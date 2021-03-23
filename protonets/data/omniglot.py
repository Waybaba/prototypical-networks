import os
import sys
import glob

from functools import partial

import numpy as np
from PIL import Image

import torch
from torchvision.transforms import ToTensor

from torchnet.dataset import ListDataset, TransformDataset
from torchnet.transform import compose

import protonets
from protonets.data.base import convert_dict, CudaTransform, EpisodicBatchSampler, SequentialBatchSampler, ODGSampler

OMNIGLOT_DATA_DIR  = os.path.join(os.path.dirname(__file__), '../../data/omniglot')
OMNIGLOT_CACHE = { }

def load_image_path(key, out_field, d):
    d[out_field] = Image.open(d[key])
    return d

def convert_tensor(key, d):
    d[key] = 1.0 - torch.from_numpy(np.array(d[key], np.float32, copy=False)).transpose(0, 1).contiguous().view(1, d[key].size[0], d[key].size[1])
    return d # {data: tesnor, file_name: xxxx, class: }

def rotate_image(key, rot, d):
    d[key] = d[key].rotate(rot)
    return d

def scale_image(key, height, width, d):
    d[key] = d[key].resize((height, width))
    return d

def load_class_images(d):
    '''
    input: class name. eg. Atemayar_Qelisayer/c...r12/rot180
    return: eg. {'class': 'Aurek-Besh/character21/rot180', 'data': [20, 1, 28, 28]} # include all data of 
    '''
    if d['class'] not in OMNIGLOT_CACHE:
        alphabet, character, rot = d['class'].split('/')
        image_dir = os.path.join(OMNIGLOT_DATA_DIR, 'data', alphabet, character)

        class_images = sorted(glob.glob(os.path.join(image_dir, '*.png'))) # get all path in dir Atemayar_Qelisayer/c...r12/rot180
        if len(class_images) == 0:
            raise Exception("No images found for omniglot class {} at {}. Did you run download_omniglot.sh first?".format(d['class'], image_dir))

        image_ds = TransformDataset(ListDataset(class_images), # here, we use loader for coherent, actually, we can use normal file loader.
                                    compose([partial(convert_dict, 'file_name'),
                                             partial(load_image_path, 'file_name', 'data'),
                                             partial(rotate_image, 'data', float(rot[3:])), # rotate according to rot_xxx
                                             partial(scale_image, 'data', 28, 28),
                                             partial(convert_tensor, 'data')]))

        loader = torch.utils.data.DataLoader(image_ds, batch_size=len(image_ds), shuffle=False)

        for sample in loader:
            OMNIGLOT_CACHE[d['class']] = sample['data']
            break # only need one sample because batch size equal to dataset length

    return { 'class': d['class'], 'data': OMNIGLOT_CACHE[d['class']] } # {'class': 'Aurek-Besh/character21/rot180', 'data': [20, 1, 28, 28]}

def load_class_images_odg(d):
    '''
    input: class name. eg. Atemayar_Qelisayer/c...r12/rot180
    return: eg. {'class': 'Aurek-Besh/character21/rot180', 'data': [20, 1, 28, 28], 'data_next': [20, 1, 28, 28]}
    '''
    def next_rot(c):
        alphabet, character, rot = c.split('/')
        rot = rot[:3] + str(int(rot[3:]) + 15)
        c = '/'.join([alphabet, character, rot])
        return c
    origin_class = d['class']
    next_class = next_rot(origin_class)
    for c in [origin_class, next_class]:
        if c not in OMNIGLOT_CACHE:
            alphabet, character, rot = c.split('/')
            image_dir = os.path.join(OMNIGLOT_DATA_DIR, 'data', alphabet, character)

            class_images = sorted(glob.glob(os.path.join(image_dir, '*.png'))) # get all path in dir Atemayar_Qelisayer/c...r12/rot180
            if len(class_images) == 0:
                raise Exception("No images found for omniglot class {} at {}. Did you run download_omniglot.sh first?".format(c, image_dir))

            image_ds = TransformDataset(ListDataset(class_images), # here, we use loader for coherent, actually, we can use normal file loader.
                                        compose([partial(convert_dict, 'file_name'),
                                                partial(load_image_path, 'file_name', 'data'),
                                                partial(rotate_image, 'data', float(rot[3:])), # rotate according to rot_xxx
                                                partial(scale_image, 'data', 28, 28),
                                                partial(convert_tensor, 'data')]))

            loader = torch.utils.data.DataLoader(image_ds, batch_size=len(image_ds), shuffle=False)

            for sample in loader:
                OMNIGLOT_CACHE[c] = sample['data']
                break # only need one sample because batch size equal to dataset length
    return {'class': origin_class, 'data': OMNIGLOT_CACHE[origin_class], 'data_next': OMNIGLOT_CACHE[next_class]} # {'class': 'Aurek-Besh/character21/rot180', 'data': [20, 1, 28, 28], 'data_next': [20, 1, 28, 28]}

def load_class_images_odg_unfair(mode, d):
    '''
    input: class name. eg. Atemayar_Qelisayer/c...r12/rot180
    return: eg. {'class': 'Aurek-Besh/character21/rot180', 'data': [20, 1, 28, 28], 'data_next': [20, 1, 28, 28]}
    '''
    def next_rot(c):
        alphabet, character, rot = c.split('/')
        rot = rot[:3] + "0" + str(int(rot[3:]) + 15)
        c = '/'.join([alphabet, character, rot])
        return c
    origin_class = d['class']
    next_class = next_rot(origin_class) if mode == 'odg_test' else origin_class
    for c in [origin_class, next_class]:
        if c not in OMNIGLOT_CACHE:
            alphabet, character, rot = c.split('/')
            image_dir = os.path.join(OMNIGLOT_DATA_DIR, 'data', alphabet, character)

            class_images = sorted(glob.glob(os.path.join(image_dir, '*.png'))) # get all path in dir Atemayar_Qelisayer/c...r12/rot180
            if len(class_images) == 0:
                raise Exception("No images found for omniglot class {} at {}. Did you run download_omniglot.sh first?".format(c, image_dir))

            image_ds = TransformDataset(ListDataset(class_images), # here, we use loader for coherent, actually, we can use normal file loader.
                                        compose([partial(convert_dict, 'file_name'),
                                                partial(load_image_path, 'file_name', 'data'),
                                                partial(rotate_image, 'data', float(rot[3:])), # rotate according to rot_xxx
                                                partial(scale_image, 'data', 28, 28),
                                                partial(convert_tensor, 'data')]))

            loader = torch.utils.data.DataLoader(image_ds, batch_size=len(image_ds), shuffle=False)

            for sample in loader:
                OMNIGLOT_CACHE[c] = sample['data']
                break # only need one sample because batch size equal to dataset length
    return {'class': origin_class, 'data': OMNIGLOT_CACHE[origin_class], 'data_next': OMNIGLOT_CACHE[next_class]} # {'class': 'Aurek-Besh/character21/rot180', 'data': [20, 1, 28, 28], 'data_next': [20, 1, 28, 28]}

def extract_episode(n_support, n_query, d):
    # data: N x C x H x W
    n_examples = d['data'].size(0) # = 20

    if n_query == -1: # -1 means all except for n_support
        n_query = n_examples - n_support

    example_inds = torch.randperm(n_examples)[:(n_support+n_query)]
    support_inds = example_inds[:n_support]
    query_inds = example_inds[n_support:]

    xs = d['data'][support_inds]
    xq = d['data'][query_inds]

    return { # randoms split into support set and query set
        'class': d['class'],
        'xs': xs,
        'xq': xq
    }

def extract_episode_odg(n_support, n_query, d):
    # data: N x C x H x W
    n_examples = d['data'].size(0) # = 20

    if n_query == -1: # -1 means all except for n_support
        n_query = n_examples - n_support

    example_inds = torch.randperm(n_examples)[:(n_support+n_query)]
    support_inds = example_inds[:n_support]
    query_inds = example_inds[n_support:]

    xs = d['data'][support_inds]
    xq = d['data_next'][query_inds] # note here

    return { # randoms split into support set and query set
        'class': d['class'],
        'xs': xs,
        'xq': xq
    }

def load(opt, splits):
    split_dir = os.path.join(OMNIGLOT_DATA_DIR, 'splits', opt['data.split'])

    ret = { }
    for split in splits:
        if split in ['val', 'test', 'odg_test'] and opt['data.test_way'] != 0: # get way value # TODO odg_test, odg_train
            n_way = opt['data.test_way']
        else:
            n_way = opt['data.way']

        if split in ['val', 'test', 'odg_test'] and opt['data.test_shot'] != 0: # get shot value
            n_support = opt['data.test_shot']
        else:
            n_support = opt['data.shot']

        if split in ['val', 'test', 'odg_test'] and opt['data.test_query'] != 0: # get query value
            n_query = opt['data.test_query']
        else:
            n_query = opt['data.query']

        if split in ['val', 'test', 'odg_test']:
            n_episodes = opt['data.test_episodes'] # get epis value
        else:
            n_episodes = opt['data.train_episodes']
        # train: n_episodes: 100, n_query:5 , n_support:5, n_way: 60
        # val:   n_episodes: 100, n_query:15, n_support:5, n_way: 60
        transforms = [partial(convert_dict, 'class'), # convert to {class: "Angelic/character01/rot000"}
                    #   load_class_images, # eg. {'class': 'Aurek-Besh/character21/rot180', 'data': [20, 1, 28, 28]} # include all data of \
                    #   partial(extract_episode, n_support, n_query)], # split into support query # {'class': , 'xs': , 'xq': }
                      load_class_images_odg, # {'class': 'Aurek-Besh/character21/rot180', 'data': [20, 1, 28, 28], 'data_next': [20, 1, 28, 28]}
                    #   partial(load_class_images_odg_unfair, split), # TODO unfair ordg
                      partial(extract_episode_odg, n_support, n_query)] # split into support query # {'class': , 'xs': , 'xq': }
        if opt['data.cuda']: # append cuda transformer
            transforms.append(CudaTransform())

        transforms = compose(transforms)

        class_names = [] # list of data paths
        with open(os.path.join(split_dir, "{:s}.txt".format(split)), 'r') as f:
            for class_name in f.readlines():
                class_names.append(class_name.rstrip('\n')) # 'Angelic/character01/rot000'
        ds = TransformDataset(ListDataset(class_names), transforms) # input item: eg. 'Angelic/character01/rot000'. output: eg. {'class': , 'xs': , 'xq': } # each items is one class

        if 'odg' in split:
            sampler = ODGSampler(len(ds), n_way, n_episodes, split) # split can be 'odg_train' 'odg_test'
        elif opt['data.sequential']:
            sampler = SequentialBatchSampler(len(ds))
        else: # default
            sampler = EpisodicBatchSampler(len(ds), n_way, n_episodes) # random sample n_way classes

        # use num_workers=0, otherwise may receive duplicate episodes
        ret[split] = torch.utils.data.DataLoader(ds, batch_sampler=sampler, num_workers=0) # ['train', 'val']

    return ret
