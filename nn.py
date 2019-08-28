# Copyright (C) 2019 Willy Po-Wei Wu & Elvis Yu-Jing Lin <maya6282@gmail.com, elvisyjlin@gmail.com>
# 
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.


import random
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from switchable_norm import SwitchNorm2d
from torchsummary import summary


def weights_init(m):
    if isinstance(m, nn.Conv2d):  # glorot_uniform weight, zero bias
        nn.init.xavier_uniform_(m.weight.data)
        m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):  # he_normal weight, zero bias
        nn.init.kaiming_normal_(m.weight.data)
        m.bias.data.zero_()
    else:
        pass

def tile_like(x, target):  # tile_size = 256 or 4
    x = x.view(x.size(0), x.size(1), 1, 1)
    x = x.repeat(1, 1, target.size(2), target.size(3))
    return x

def count_trainable_parameters(model):
    return sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])

class ResidualBlock(nn.Module):
    def __init__(self, n_in, n_out, kernel_size):
        super(ResidualBlock, self).__init__()
        self.f = nn.Sequential(
            nn.Conv2d(n_in, n_out, kernel_size=kernel_size, padding=1, stride=1), 
            SwitchNorm2d(n_out, momentum=0.9), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(n_out, n_out, kernel_size=kernel_size, padding=1, stride=1), 
            SwitchNorm2d(n_out, momentum=0.9), 
        )
    def forward(self, x):
        return x + self.f(x)

def lsgan(xs, ts):
    loss = torch.zeros_like(xs[0])
    for x, t in zip(xs, ts):
        target = torch.ones_like(x) * t
        loss += ((target - x) ** 2)  # L2 Loss
    return loss

class GAN(nn.Module):
    def __init__(self, args):
        print('Building GAN...')
        super(GAN, self).__init__()
        self.lr = args.lr
        self.betas = (args.b1, args.b2)
        self.batch_size = args.batch_size
        self.repeat_G = args.repeat_G
        self.lambda1 = args.l1
        self.lambda2 = args.l2
        self.lambda3 = args.l3
        self.lambda4 = args.l4
        self.gamma = args.gamma
        self.decay = self.lr / args.steps
        self.latent_size = len(args.selected_attributes)
        self.gpu = args.gpu
        self.multi_gpu = args.multi_gpu
        
        self.zero_consistency = args.zero_consistency
        self.cycle_consistency = args.cycle_consistency
        self.interpolation_regularize = args.interpolation_regularize
        self.orthogonal_regularize = args.orthogonal_regularize
        
        self.G = G(3, self.latent_size, self.repeat_G)
        self.D = D(3, self.latent_size)
        self.G.apply(weights_init)
        self.D.apply(weights_init)
        if self.gpu:
            self.G.cuda()
            self.D.cuda()
        if self.multi_gpu:
            self.G = nn.DataParallel(self.G)
            self.D = nn.DataParallel(self.D)
            print('Using', torch.cuda.device_count(), 'GPUs!')
            
        self.optimG = optim.Adam(
            self.G.parameters(), lr=self.lr, 
            betas=self.betas, weight_decay=self.decay
        )
        self.optimD = optim.Adam(
            self.D.parameters(), lr=self.lr, 
            betas=self.betas, weight_decay=self.decay
        )
        
    def train_G(self, img_a, img_b, z_ab):
        img_a2b = self.G(img_a, z_ab)
        img_a2a = self.G(img_a, torch.zeros_like(z_ab))
        img_a2b2a = self.G(img_a2b, -z_ab)
        
        alpha_i = torch.rand((z_ab.size(0), 1))
        alpha_i = alpha_i.cuda() if self.gpu else alpha_i
        img_a2bi = self.G(img_a, alpha_i * z_ab)
        
        d_real, dc_real = self.D(img_a, img_b, z_ab)
        d_fake, dc_fake = self.D(img_a, img_a2b, z_ab)
        di_inte = self.D(img_a2bi, critic=True)
        
        gf_loss = lsgan([d_real, d_fake], [0, 1]).mean()
        gc_loss = lsgan([dc_real, dc_fake], [0, 1]).mean()
        
        g_loss_rec1 = (img_a - img_a2b2a).abs().mean() if self.cycle_consistency else torch.zeros(1)
        g_loss_rec2 = (img_a - img_a2a).abs().mean() if self.zero_consistency else torch.zeros(1)
        gr_loss = self.lambda1 * g_loss_rec1 + self.lambda2 * g_loss_rec2
        
        if self.interpolation_regularize:
            gi_loss = di_inte ** 2
            gi_loss = gi_loss.mean()
        else:
            gi_loss = torch.zeros(1)
            gi_loss = gi_loss.cuda() if self.gpu else gi_loss
        
        if self.orthogonal_regularize:
            go_loss = None
            for p in self.G.parameters():
                if len(p.size()) == 4 and p.size(3) > 1:
                    for kw in range(p.size(2)):
                        for kh in range(p.size(3)):
                            w = torch.mm(p[:,:,kw,kh].transpose(0, 1), p[:,:,kw,kh])
                            m = 1 - torch.eye(*w.size(), device=w.device)
                            a = w * m
                            a = torch.mm(a.transpose(0, 1), a)
                            tr = torch.sum(a * torch.eye(*a.size(), device=a.device))
                            go_loss = tr if go_loss is None else go_loss + tr
        else:
            go_loss = torch.zeros(1)
            go_loss = go_loss.cuda() if self.gpu else go_loss
            
        g_loss = gf_loss + gc_loss + gr_loss + self.lambda3 * gi_loss + self.lambda4 * go_loss
        
        self.optimG.zero_grad()
        g_loss.backward()
        self.optimG.step()
        
        errG = {
            'g_loss': g_loss.item(), 
            'gf_loss': gf_loss.item(), 
            'gc_loss': gc_loss.item(), 
            'gr_loss': gr_loss.item(), 
            'gi_loss': gi_loss.item(), 
            'go_loss': go_loss.item()
        }
        return errG
    
    def train_D(self, img_a, img_b, img_c, z_ab, z_ac, z_cb):
        img_a2b = self.G(img_a, z_ab).detach()
        img_a2a = self.G(img_a, torch.zeros_like(z_ab)).detach()
        
        alpha_i = torch.rand((z_ab.size(0), 1))
        alpha_i = alpha_i.cuda() if self.gpu else alpha_i
        img_a2bi = self.G(img_a, alpha_i * z_ab)
        
        d_real, dc_real  = self.D(img_a, img_b, z_ab)
        d_fake, dc_fake = self.D(img_a, img_a2b, z_ab)
        d_w_ori, dc_w_ori = self.D(img_c, img_b, z_ab)
        d_w_tar, dc_w_tar = self.D(img_a, img_c, z_ab)
        d_w_vec1, dc_w_vec1 = self.D(img_a, img_b, z_ac)
        d_w_vec2, dc_w_vec2 = self.D(img_a, img_b, z_cb)
        di_real = self.D(img_a2a, critic=True)
        di_fake = self.D(img_a2b, critic=True)
        di_inte = self.D(img_a2bi, critic=True)
        
        df_loss = lsgan([d_real, d_fake], [1, 0]).mean()
        dc_loss = lsgan(
            [dc_real, dc_fake, dc_w_ori, dc_w_tar, dc_w_vec1, dc_w_vec2], 
            [1, 0, 0, 0, 0, 0]
        ).mean()
        
        """
        # The following part is the original gradient penalty.
        # We mix the images A and A2B, instead of B and A2B, to compute D's gradients.
        #
        # However, since PyTorch 1.0.0, AutoGrad is deleted once the 
        # function exits when the tensors are distributed among multi GPUs (nn.Parallel).
        # See the issue (https://github.com/pytorch/pytorch/issues/16532).
        #
        # Before the bug is fixed, we have to compute GP inline with the backward function.
        
        def cal_df_gp(img_a, img_a2b, z_ab):
            alpha = torch.rand(img_a.size(0), 1, 1, 1)
            alpha = alpha.cuda(async=self.multi_gpu) if self.gpu else alpha
            mix_tar = (alpha * img_a + (1 - alpha) * img_a2b).requires_grad_(True)  # interpolates
            mix_outputs, _ = self.D(img_a, mix_tar, z_ab)
            gradients = autograd.grad(
                outputs=mix_outputs, inputs=mix_tar,
                grad_outputs=torch.ones_like(mix_outputs),
                create_graph=True, retain_graph=True, only_inputs=True
            )[0]
            gradients = gradients.view(gradients.size(0), -1)
            gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
            return gradient_penalty
        df_gp = cal_df_gp(img_a, img_a2b, z_ab)
        """
        
        # Inline gradient penalty
        alpha = torch.rand(img_a.size(0), 1, 1, 1)
        alpha = alpha.cuda(async=self.multi_gpu) if self.gpu else alpha
        mix_tar = (alpha * img_a + (1 - alpha) * img_a2b).requires_grad_(True)  # interpolates
        mix_outputs, _ = self.D(img_a, mix_tar, z_ab)
        gradients = autograd.grad(
            outputs=mix_outputs, inputs=mix_tar,
            grad_outputs=torch.ones_like(mix_outputs),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        df_gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        if self.interpolation_regularize:
            real_mask = (alpha_i <  0.5).type(torch.float32)
            di_loss = (  real_mask) * ((  alpha_i-di_inte) ** 2 + di_real ** 2) + \
                      (1-real_mask) * ((1-alpha_i-di_inte) ** 2 + di_fake ** 2)
            di_loss = di_loss.mean()
        else:
            di_loss = torch.zeros(1)
            di_loss = di_loss.cuda() if self.gpu else di_loss
        
        d_loss = df_loss + dc_loss + 150 * df_gp + self.lambda3 * di_loss
        
        self.optimD.zero_grad()
        d_loss.backward()
        self.optimD.step()
        
        errD = {
            'd_loss': d_loss.item(), 
            'df_loss': df_loss.item(), 
            'dc_loss': dc_loss.item(), 
            'df_gp': df_gp.item(), 
            'di_loss': di_loss.item()
        }
        return errD
    
    def train(self):
        self.G.train()
        self.D.train()
    
    def eval(self):
        self.G.eval()
        self.D.eval()
    
    def summary(self):
        G = self.G.module if self.multi_gpu else self.G
        D = self.D.module if self.multi_gpu else self.D
        # print(self.G)
        print('# of trainable parameters in G:', count_trainable_parameters(G))
        g = summary(G, [(3, 256, 256), (self.latent_size, )], 
                    dtype=[torch.float, torch.float], 
                    use_gpu=self.gpu, return_str=True)
        # print(self.D)
        print('# of trainable parameters in D:', count_trainable_parameters(D))
        d = summary(D, [(3, 256, 256), (3, 256, 256), (self.latent_size, )], 
                    dtype=[torch.float, torch.float, torch.float], 
                    use_gpu=self.gpu, return_str=True)
        # lambda *x: self.D(*x, critic=True)
        d_critic = summary(D, [(3, 256, 256)], 
                           dtype=[torch.float], 
                           use_gpu=self.gpu, return_str=True, forward_fn=lambda x: D(x, critic=True))
        return g, d, d_critic
        
    def save(self, file):
        print('Saving weights to', file, '...')
        torch.save({
            'G': self.G.state_dict(), 
            'D': self.D.state_dict(), 
            'optimG': self.optimG.state_dict(), 
            'optimD': self.optimD.state_dict()
        }, file)
        
    def load(self, file):
        print('Loading saved weights from', file, '...')
        states = torch.load(file, map_location=lambda storage, loc: storage)
        if 'G' in states:
            self.G.load_state_dict(states['G'])
        if 'D' in states:
            self.D.load_state_dict(states['D'])
        if 'optimG' in states:
            self.optimG.load_state_dict(states['optimG'])
        if 'optimD' in states:
            self.optimD.load_state_dict(states['optimD'])
        
class G(nn.Module):
    def __init__(self, n_c, n_z, n_repeat=6):
        print('Building generator...')
        super(G, self).__init__()
        self.n_z = n_z
        
        self.conv_in = nn.Sequential(
            nn.Conv2d(n_c + n_z, 64, kernel_size=7, padding=3, stride=1), 
            SwitchNorm2d(64, momentum=0.9), 
            nn.ReLU(inplace=True), 
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, padding=1, stride=2), 
            SwitchNorm2d(128, momentum=0.9), 
            nn.ReLU(inplace=True), 
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, padding=1, stride=2), 
            SwitchNorm2d(256, momentum=0.9), 
            nn.ReLU(inplace=True), 
        )
        resb_layers = [ResidualBlock(256, 256, 3) for _ in range(n_repeat)]
        self.resb = nn.Sequential(*resb_layers)
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, padding=1, stride=2), 
            SwitchNorm2d(128, momentum=0.9), 
            nn.ReLU(inplace=True), 
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, padding=1, stride=2), 
            SwitchNorm2d(64, momentum=0.9), 
            nn.ReLU(inplace=True), 
        )
        self.conv_out = nn.Sequential(
            nn.Conv2d(64, n_c, kernel_size=7, padding=3, stride=1), 
            nn.Tanh(), 
        )
    def forward(self, img, z):
        tiled_z = tile_like(z, img)
        x = torch.cat([img, tiled_z], dim=1)
        h = self.conv_in(x)
        h = self.down1(h)
        h = self.down2(h)
        h = self.resb(h)
        h = self.up2(h)
        h = self.up1(h)
        y = self.conv_out(h)
        return y
    
class D(nn.Module):
    def __init__(self, n_c, n_z, n_filters=[64, 128, 256, 512, 1024, 2048]):
        print('Building discriminator...')
        super(D, self).__init__()
        layers = []
        n_in = n_c
        for n_f in n_filters:
            layers += [nn.Conv2d(n_in, n_f, kernel_size=4, padding=1, stride=2)]
            layers += [nn.LeakyReLU(negative_slope=0.01, inplace=True)]
            n_in = n_f
        self.convs = nn.Sequential(*layers)
        self.conv_adv = nn.Conv2d(n_in, 1, kernel_size=1, padding=0, stride=1)
        self.conv_int = nn.Conv2d(n_in, 16, kernel_size=1, padding=0, stride=1)
        n_in_c = n_filters[-1] * 2 + n_z
        self.convs_cls = nn.Sequential(
            nn.Conv2d(n_in_c, 2048, kernel_size=1, padding=0, stride=1), 
            nn.LeakyReLU(negative_slope=0.01, inplace=True), 
            nn.Conv2d(2048, 1, kernel_size=1, padding=0, stride=1), 
        )
    def forward(self, img_a, img_b=None, z=None, critic=False):
        if not critic:
            assert img_a is not None and img_b is not None and z is not None
            h_a = self.convs(img_a)
            h_b = self.convs(img_b)
            y_1 = self.conv_adv(h_b)
            tiled_z = tile_like(z, h_a)
            h = torch.cat([h_a, h_b, tiled_z], dim=1)
            y_2 = self.convs_cls(h)
            return y_1, y_2
        else:
            assert img_a is not None
            h = self.convs(img_a)
            h = self.conv_int(h)
            y = h.view(h.size(0), -1).mean(1, keepdim=True)  # Global average pooling
            return y