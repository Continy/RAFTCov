# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

import os
import math
import random
from glob import glob
import os.path as osp

from utils import frame_utils
from utils.augmentor import FlowAugmentor, SparseFlowAugmentor
import re


class FlowDataset(data.Dataset):

    def __init__(self, aug_params=None, sparse=False, npy=False):
        self.augmentor = None
        self.sparse = sparse
        self.npy = npy
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.depth_list = []
        self.image_list = []
        self.extra_info = []

    def __getitem__(self, index):

        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        valid = None
        if self.sparse:
            flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
        elif self.npy == False:
            flow = frame_utils.read_gen(self.flow_list[index])

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])
        if self.npy == True:
            flow = np.load(self.flow_list[index])
        else:
            flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(
                    img1, img2, flow, valid)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        if len(flow.shape) == 3:  #flow
            flow = torch.from_numpy(flow).permute(2, 0, 1).float()
            if valid is not None:
                valid = torch.from_numpy(valid).float()
            else:
                valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)
                valid = valid.float()
            return img1, img2, flow, valid
        elif len(flow.shape) == 2:  #depth
            flow = torch.from_numpy(flow).unsqueeze(0).float()
            return img1, img2, flow

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self

    def __len__(self):
        return len(self.image_list)


class TartanAir(FlowDataset):

    def __init__(
            self,
            aug_params=None,
            root='D:\\gits\\FlowFormer-Official\\datasets\\abandonedfactory\\Easy\\',
            stereo=False,
            traj=['abandonedfactory']):
        super(TartanAir, self).__init__(aug_params, npy=True)

        if stereo:
            dirnames = traj
            for path in dirnames:
                datalist = os.listdir(os.path.join(root, path, 'Data'))
                pattern = re.compile(r'P00\d')
                Plist = []
                for i in datalist:
                    if pattern.match(i):
                        Plist.append(i)
                for i in Plist:
                    depth_length = len(
                        glob(
                            os.path.join(root, path, 'Data', i, 'depth_left',
                                         '*_depth.npy')))
                    print('find {} depth files in {}'.format(
                        depth_length,
                        os.path.join(root, path, 'Data', i, 'depth_left')))
                    depth = sorted(
                        glob(
                            os.path.join(root, path, 'Data', i, 'depth_left',
                                         '*_depth.npy')))
                    #print(flows)
                    imageL = sorted(
                        glob(
                            os.path.join(root, path, 'Data', i, 'image_left',
                                         '*.png')))
                    imageR = sorted(
                        glob(
                            os.path.join(root, path, 'Data', i, 'image_right',
                                         '*.png')))
                    for i in range(depth_length - 1):
                        self.flow_list += [depth[i]]
                        self.image_list += [[imageL[i], imageR[i]]]
        else:

            dirnames = traj
            for path in dirnames:
                datalist = os.listdir(os.path.join(root, path, 'Data'))
                pattern = re.compile(r'P00\d')
                Plist = []
                for i in datalist:
                    if pattern.match(i):
                        Plist.append(i)
                for i in Plist:
                    flow_length = len(
                        glob(
                            os.path.join(root, path, 'Data', i, 'flow',
                                         '*_flow.npy')))
                    print('find {} flow files in {}'.format(
                        flow_length, os.path.join(root, path, 'Data', i,
                                                  'flow')))
                    flows = sorted(
                        glob(
                            os.path.join(root, path, 'Data', i, 'flow',
                                         '*_flow.npy')))
                    #print(flows)
                    images = sorted(
                        glob(
                            os.path.join(root, path, 'Data', i, 'image_left',
                                         '*.png')))
                    for i in range(flow_length - 1):
                        self.flow_list += [flows[i]]
                        self.image_list += [[images[i], images[i + 1]]]


class FixedSizeDataset(data.Dataset):

    def __init__(self, dataset, size):
        self.dataset = dataset
        self.size = size

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return min(len(self.dataset), self.size)


def fetch_dataloader(args):
    """ Create the data loader for the corresponding trainign set """
    if args.stage == 'tartanair':
        if not args.stereo:
            aug_params = {
                'crop_size': args.image_size,
                'min_scale': -0.1,
                'max_scale': 0.3,
                'do_flip': False
            }
        else:
            aug_params = None
        if args.root != None:
            train_dataset = TartanAir(aug_params,
                                      root=args.root,
                                      stereo=args.stereo,
                                      traj=args.traj)
        else:
            train_dataset = TartanAir(aug_params, stereo=args.stereo)
    else:

        raise ValueError(
            'This dataset is not supported, please check core/datasets.py and add the dataset'
        )

    if args.folderlength is not None:
        train_dataset = FixedSizeDataset(train_dataset, args.folderlength)
        train_loader = data.DataLoader(train_dataset,
                                       batch_size=args.batch_size,
                                       pin_memory=True,
                                       shuffle=False,
                                       num_workers=0,
                                       drop_last=True)
    else:
        train_loader = data.DataLoader(train_dataset,
                                       batch_size=args.batch_size,
                                       pin_memory=True,
                                       shuffle=True,
                                       num_workers=0,
                                       drop_last=True)
    print('Training with %d image pairs' % len(train_dataset))
    return train_loader
