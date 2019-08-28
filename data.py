# Copyright (C) 2019 Willy Po-Wei Wu & Elvis Yu-Jing Lin <maya6282@gmail.com, elvisyjlin@gmail.com>
# 
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.


import numpy as np
import skimage.io as io
import torch
import torchvision.transforms as transforms
from os.path import join


class CelebA(object):
    def __init__(self, path, image_size, selected_attrs=None, 
                 filter_attrs={}, mode='train', test_num=2000):
        assert mode in ['train', 'val'], 'Unsupported mode: {}'.format(mode)
        self.path = path
        print('Loading annotations...')
        self.annotations, self.selected_attrs = load_annotations(join(path, 'list_attr_celeba.txt'), selected_attrs)
        print('Loading image list...')
        self.image_list = list(sorted(self.annotations.keys()))
        self.filter(filter_attrs)
        if mode == 'train':
            self.tf = transforms.Compose([
                transforms.ToPILImage(), 
                transforms.Resize(image_size), 
                transforms.CenterCrop(image_size), 
                transforms.RandomHorizontalFlip(p=0.5), 
                transforms.ToTensor(), 
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        if mode == 'val':
            self.tf = transforms.Compose([
                transforms.ToPILImage(), 
                transforms.Resize(image_size), 
                transforms.CenterCrop(image_size), 
                transforms.ToTensor(), 
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        print('Splitting image list...')
        if test_num > -1:
            if mode == 'train':
                print('Picking training images')
                self.image_list = self.image_list[test_num:]
            if mode == 'val':
                print('Picking testing images')
                self.image_list = self.image_list[:test_num]
        print('CelebA dataset loaded.')
    def get(self, index):
        img = io.imread(join(self.path, 'celeba', self.image_list[index]))
        att = self.annotations[self.image_list[index]]
        return self.tf(img), torch.tensor(att)
    def __len__(self):
        return len(self.image_list)
    def filter(self, attributes):
        to_remove = []
        for img_idx, img in enumerate(self.image_list):
            for attr, val in attributes.items():
                attr_idx = self.selected_attrs.index(attr)
                if self.annotations[img][attr_idx] == val:
                    to_remove.append(img_idx)
                    break
        for img_idx in reversed(to_remove):
            del self.image_list[img_idx]
            del self.annotations[img_idx]

class CelebAHQ(object):
    def __init__(self, path, image_size, selected_attrs=None, 
                 filter_attrs={}, mode='train', test_num=2000):
        assert mode in ['train', 'val'], 'Unsupported mode: {}'.format(mode)
        self.path = path
        self.image_size = image_size
        print('Loading annotations...')
        self.annotations, self.selected_attrs = load_annotations(join(path, 'list_attr_celeba.txt'), selected_attrs)
        print('Loading image list...')
        self.image_list = load_image_list(join(path, 'image_list.txt'))
        self.filter(filter_attrs)
        if mode == 'train':
            self.tf = transforms.Compose([
                transforms.ToPILImage(), 
                transforms.Resize(image_size), 
                transforms.CenterCrop(image_size), 
                transforms.RandomHorizontalFlip(p=0.5), 
                transforms.ToTensor(), 
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        if mode == 'val':
            self.tf = transforms.Compose([
                transforms.ToPILImage(), 
                transforms.Resize(image_size), 
                transforms.CenterCrop(image_size), 
                transforms.ToTensor(), 
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        print('Splitting image list...')
        if test_num > -1:
            if mode == 'train':
                print('Picking training images')
                #self.image_list = self.image_list[test_num:]
                #self.length = self.length - test_num
                
                # Pick all images as training set
                self.image_list = self.image_list
            if mode == 'val':
                print('Picking testing images')
                self.image_list = self.image_list[:test_num]
        print('CelebA-HQ dataset loaded.')
    def get(self, index):
        img = io.imread(join(self.path, 'celeba-hq/celeba-{:d}'.format(self.image_size), '{:d}.jpg'.format(index)))
        att = self.annotations[self.image_list[index]]
        return self.tf(img), torch.tensor(att)
    def __len__(self):
        return len(self.image_list)
    def filter(self, attributes):
        to_remove = []
        for img_idx, img in enumerate(self.image_list):
            for attr, val in attributes.items():
                attr_idx = self.selected_attrs.index(attr)
                if self.annotations[self.image_list[img_idx]][attr_idx] == val:
                    to_remove.append(img_idx)
                    break
        for img_idx in reversed(to_remove):
            del self.image_list[img_idx]
            del self.annotations[img_idx]

class PairedData(object):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = mode == 'train'
        self.i = 0
    def next(self, gpu=False, multi_gpu=False):
        if self.shuffle:
            idxs = np.random.choice(len(self.dataset), self.batch_size)
        else:
            idxs = list(range(self.i, self.i + self.batch_size))
            self.i = self.i + self.batch_size
            if self.i + self.batch_size >= len(self):
                self.i = 0
        
        imgs = [None] * self.batch_size
        atts = [None] * self.batch_size
        for i in range(len(idxs)):
            img, att = self.dataset.get(idxs[i])
            imgs[i] = img
            atts[i] = att
        imgs = torch.stack(imgs)
        atts = torch.stack(atts)
        if gpu:
            imgs = imgs.cuda(async=multi_gpu)
            atts = atts.cuda(async=multi_gpu)
        return imgs, atts
    def __len__(self):
        return len(self.dataset)

def load_annotations(file, selected_attrs=None):
    lines = open(file).readlines()
    '''
    202599
    Attribute names
    000001.jpg -1  1  1 -1 -1 -1 -1 -1 -1 -1 -1  1 -1 -1 -1 -1 -1 -1  1  1 -1  1 -1 -1  1 -1 -1  1 -1 -1 -1  1  1 -1  1 -1  1 -1 -1  1
    ...
    
    selected_attrs = ['5_o_Clock_Shadow', 'Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Male', 'Mustache', 'Pale_Skin', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Hat', 'Young']
    '''
    attrs = lines[1].split()
    if selected_attrs is None:
        selected_attrs = attrs
    selected_attrs_idx = [attrs.index(a) for a in selected_attrs]

    annotations = {}
    for line in lines[2:]:
        tokens = line.split()
        file = tokens[0]
        anno = [(int(t)+1)/2 for t in tokens[1:]]
        anno = [anno[idx] for idx in selected_attrs_idx]
        annotations[file] = anno
    return annotations, selected_attrs

def load_image_list(file):
    lines = open(file).readlines()[1:]
    '''
    idx         orig_idx    orig_file   proc_md5                          final_md5

    0           119613      119614.jpg  0be7e162e25c06f50dd5c1090007f2cf  d76ed3e87c8bc20f82757a2dd95026ba

    1           99094       099095.jpg  1e2d301e9b3d1b64b2e560243b5c109c  c391ae358c1a00e715982050b6446109
    '''
    image_list = [None] * len(lines)
    for line in lines:
        tokens = line.split()
        idx = int(tokens[0])
        file = tokens[2]
        image_list[idx] = file
    return image_list