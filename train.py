# Copyright (C) 2019 Willy Po-Wei Wu & Elvis Yu-Jing Lin <maya6282@gmail.com, elvisyjlin@gmail.com>
# 
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.


# Attributes:
# '5_o_Clock_Shadow', 'Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Male', 'Mustache', 'Pale_Skin', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Hat', 'Young'

import argparse
import numpy as np
import os
import time
import yaml

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter

from helpers import add_scalar_dict
from nn import GAN
from ops import inv_normalize


def parse(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='data path')
    parser.add_argument('--dataset', type=str, help='dataset', default='celeba-hq', choices=['celeba-hq', 'celeba', 'wikiart-genre+style'])
    parser.add_argument('--config', type=str, help='config file')
    parser.add_argument('--load', type=str, help='load model from file')
    parser.add_argument('--save', type=str, default='model/weights.pth', help='save model as file')
    parser.add_argument('--steps', type=int, default=400000, help='training steps')
    parser.add_argument('--start_step', type=int, default=0, help='start training step')
    parser.add_argument('--name', type=str, help='experiment name')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--multi_gpu', action='store_true')
    
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--b1', type=float, default=0.5, help='beta 1')
    parser.add_argument('--b2', type=float, default=0.999, help='beta 2')
    parser.add_argument('--iterG', type=int, default=1, help='iterations of generator')
    parser.add_argument('--iterD', type=int, default=1, help='iterations of discriminator')
    parser.add_argument('--batch_size', type=int, default=6, help='batch size')
    parser.add_argument('--l1', type=float, default=10.0, help='lambda 1')
    parser.add_argument('--l2', type=float, default=10.0, help='lambda 2')
    parser.add_argument('--l3', type=float, default=1.0, help='lambda 3')
    parser.add_argument('--l4', type=float, default=0.0001, help='lambda 4')
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma')
    parser.add_argument('--image_size', type=int, default=256, help='image size')
    parser.add_argument('--attr', type=str, default='attributes.yaml', help='selected attribute file')
    parser.add_argument('--zero_consistency', type=bool, default=True)
    parser.add_argument('--cycle_consistency', type=bool, default=True)
    parser.add_argument('--interpolation_regularize', type=bool, default=True)
    parser.add_argument('--orthogonal_regularize', type=bool, default=True)
    
    parser.add_argument('--log_interval', type=int, default=100, help='interval of logging')
    parser.add_argument('--sample_interval', type=int, default=1000, help='interval of sampling images')
    parser.add_argument('--save_interval', type=int, default=10000, help='interval of saving models')
    
    return parser.parse_args() if args is None else parser.parse_args(args=args)

def load_config(config_file):
    print('Loading config file', config_file)
    with open(config_file, 'r', encoding='utf-8') as f:
        arg = yaml.load(f.read())
    return arg

# # Create a template of config file
# args = parse([])
# with open('config.example.yaml', 'w', encoding='utf-8') as f:
#     yaml.dump(args, f, default_flow_style=False)

args = parse()
if args.config is not None:
    args = load_config(args.config)

print('Training parameters:', args)
data_path = args.data
dataset = args.dataset
load_file = args.load
save_file = args.save
n_steps = args.steps
start_step = args.start_step
exp_name = args.name
batch_size = args.batch_size
image_size = args.image_size
n_iter_G = args.iterG
n_iter_D = args.iterD
log_interval = args.log_interval
sample_interval = args.sample_interval
save_interval = args.save_interval

gpu = args.gpu = args.gpu or args.multi_gpu
multi_gpu = args.multi_gpu

selected_attrs = [
    '5_o_Clock_Shadow', 'Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 
    'Brown_Hair', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Male', 'Mustache', 
    'Pale_Skin', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Hat', 'Young'
]
args.selected_attributes = yaml.load(open(args.attr, 'r', encoding='utf-8'))
selected_attributes = args.selected_attributes
assert type(selected_attributes) is list

os.makedirs('model', exist_ok=True)


# interpolating_attributes = ['Smiling', 'Young', 'Mustache']
interpolating_attributes = ['Young']
# test_attributes = [
#     ('Black_Hair', 1), ('Blond_Hair', 1), ('Brown_Hair', 1), 
#     ('Male', 1), ('Male', -1), ('Mustache', 1), ('Pale_Skin', 1), 
#     ('Smiling', 1), ('Bald', 1), ('Eyeglasses', 1), ('Young', 1), ('Young', -1)
# ]
test_attributes = [
    ('Black_Hair', 1), ('Blond_Hair', 1), ('Brown_Hair', 1), 
    ('Male', 1), ('Young', 1), ('Young', -1)
]
inter_annos = np.zeros(
    (10 * len(interpolating_attributes), len(selected_attributes)), 
    dtype=np.float32
)
for i, attr in enumerate(interpolating_attributes):
    index = selected_attributes.index(attr)
    inter_annos[np.arange(10*i, 10*i+10), index] = np.linspace(0.1, 1, 10)
test_annos = np.zeros(
    (len(test_attributes), len(selected_attributes)), 
    dtype=np.float32
)
for i, (attr, value) in enumerate(test_attributes):
    index = selected_attributes.index(attr)
    test_annos[i, index] = value

tf = transforms.Compose([
    transforms.Resize(image_size), 
    transforms.CenterCrop(image_size), 
    transforms.ToTensor(), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
test_dataset = datasets.ImageFolder(root='test_imgs', transform=tf)
test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size)


if dataset == 'celeba':
    from data import CelebA, PairedData
    train_dset = CelebA(
        data_path, image_size, selected_attr=selected_attributes, mode='train', test_num=2000
    )
    train_data = PairedData(train_dset, batch_size)
    valid_dset = CelebA(
        data_path, image_size, selected_attr=selected_attributes, mode='val', test_num=2000
    )
    valid_data = PairedData(valid_dset, batch_size)
if dataset == 'celeba-hq':
    from data import CelebAHQ, PairedData
    train_dset = CelebAHQ(
        data_path, image_size, selected_attr=selected_attributes, mode='train', test_num=2000
    )
    train_data = PairedData(train_dset, batch_size)
    valid_dset = CelebAHQ(
        data_path, image_size, selected_attr=selected_attributes, mode='val', test_num=2000
    )
    valid_data = PairedData(valid_dset, batch_size)
if dataset == 'wikiart-genre+style':
    from data import WikiArtMulti, PairedData
    train_dset = WikiArtMulti(
        data_path, ['genre', 'style'], image_size, mode='train'
    )
    train_data = PairedData(train_dset, batch_size)
    valid_dset = WikiArtMulti(
        data_path, ['genre', 'style'], image_size, mode='train'
    )
    valid_data = PairedData(valid_dset, batch_size)
print('# of Total Images:', len(train_data), '( Training:', len(train_data), '/ Validating:', len(valid_data), ')')
gan = GAN(args)
writer = SummaryWriter() if exp_name is None else SummaryWriter('run/' + exp_name)
writer.add_text('config', str(args))
g, d, d_critic = gan.summary()
writer.add_text('G', g.replace('\n', '  \n'))
writer.add_text('D', d.replace('\n', '  \n'))
writer.add_text('D_critic', d_critic.replace('\n', '  \n'))

if load_file is not None:
    gan.load(load_file)

fixed_img_a, fixed_att_a = valid_data.next(gpu, multi_gpu)
fixed_img_b, fixed_att_b = valid_data.next(gpu, multi_gpu)
writer.add_image('valid_img_a', make_grid(inv_normalize(fixed_img_a), nrow=8))
writer.add_image('valid_img_b', make_grid(inv_normalize(fixed_img_b), nrow=8))

for ite in range(start_step, start_step + n_steps):
    gan.train()
    t_start = time.time()
    img_a, att_a = train_data.next(gpu, multi_gpu)
    img_b, att_b = train_data.next(gpu, multi_gpu)
    img_c, att_c = train_data.next(gpu, multi_gpu)
    vec_ab = att_a - att_b
    vec_ac = att_a - att_c
    vec_cb = att_c - att_b
    for _ in range(n_iter_D):
        errD = gan.train_D(img_a, img_b, img_c, vec_ab, vec_ac, vec_cb)
    for _ in range(n_iter_G):
        errG = gan.train_G(img_a, img_b, vec_ab)
    t_end = time.time()
    
    if (ite+1) % log_interval == 0:
        add_scalar_dict(writer, errD, ite+1, 'D')
        add_scalar_dict(writer, errG, ite+1, 'G')
        writer.add_scalar('time', t_end - t_start, ite+1)
        print("{:9.6f} {:9.6f} | real: {:7.4f} wrong: {:7.4f} gp: {:7.4f}| fake: {:7.4f} wrong: {:7.4f} recs: {:7.4f} | time: {:.4f}".format(
            errD['d_loss'], errG['g_loss'], 
            errD['df_loss'], errD['dc_loss'], errD['df_gp'], 
            errG['gf_loss'], errG['gc_loss'], errG['gr_loss'], 
            t_end - t_start
        )) 
    if (ite+1) % sample_interval == 0:
        gan.eval()
        with torch.no_grad():
            vec_ab = fixed_att_a - fixed_att_b
            img_a2b = gan.G(fixed_img_a, vec_ab)
            img_a2a = gan.G(fixed_img_a, torch.zeros_like(vec_ab))
            img_a2b2a = gan.G(img_a2b, -vec_ab)
        writer.add_image('valid_img_a2b', make_grid(inv_normalize(img_a2b), nrow=8), ite+1)
        writer.add_image('valid_img_a2a', make_grid(inv_normalize(img_a2a), nrow=8), ite+1)
        writer.add_image('valid_img_a2b2a', make_grid(inv_normalize(img_a2b2a), nrow=8), ite+1)
        f = fixed_img_a.detach().unsqueeze(1).cpu()
        for anno in inter_annos:
            vec_inter = torch.tensor(anno, dtype=torch.float).repeat(fixed_img_a.size(0), 1)
            vec_inter = vec_inter.cuda() if gpu else vec_inter
            with torch.no_grad():
                img_inter = gan.G(fixed_img_a, vec_inter).detach()
            f = torch.cat([f, img_inter.unsqueeze(1).cpu()], dim=1)
        f = f.view(f.size(0)*f.size(1), f.size(2), f.size(3), f.size(4))
        writer.add_image('valid_img_inter', make_grid(inv_normalize(f), nrow=len(inter_annos)+1), ite+1)
        f = []
        g = []
        for x, y in test_dataloader:
            x = x.cuda() if gpu else x
            x0 = x.detach().unsqueeze(1).cpu()
            f.append(x0)
            g.append(x0)
            for anno in test_annos:
                vec_test = torch.tensor(anno, dtype=torch.float).repeat(x.size(0), 1)
                vec_test = vec_test.cuda() if gpu else vec_test
                with torch.no_grad():
                    img_test = gan.G(x, vec_test).detach()
                f[-1] = torch.cat([f[-1], img_test.unsqueeze(1).cpu()], dim=1)
            for anno in inter_annos:
                vec_inter = torch.tensor(anno, dtype=torch.float).repeat(x.size(0), 1)
                vec_inter = vec_inter.cuda() if gpu else vec_inter
                with torch.no_grad():
                    img_inter = gan.G(x, vec_inter).detach()
                g[-1] = torch.cat([g[-1], img_inter.unsqueeze(1).cpu()], dim=1)
        f = torch.cat(f)
        f = f.view(f.size(0)*f.size(1), f.size(2), f.size(3), f.size(4))
        g = torch.cat(g)
        g = g.view(g.size(0)*g.size(1), g.size(2), g.size(3), g.size(4))
        writer.add_image('test_img', make_grid(inv_normalize(f), nrow=len(test_annos)+1), ite+1)
        writer.add_image('test_img_inter', make_grid(inv_normalize(g), nrow=len(inter_annos)+1), ite+1)
    if (ite+1) % save_interval == 0:
        gan.save(save_file.replace('.pth', '.{:d}.pth'.format(ite)))