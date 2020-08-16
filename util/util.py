"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
import os
import cv2
from util.flow_util import flow2img
from os.path import basename

def tensor2im(input_image, idx=0, imtype=np.uint8):
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[idx].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile((image_numpy - 0.5) * 2, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def tensor2flow(flow, idx=0, imtype=np.uint8, max=255):

    ### transform the flow from [-1, 1] which represent the sample location
    # to [-h, h] which represent the pixel motion

    B, _, H, W = flow.size()
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    base_grid = torch.cat((yy, xx), 1).float().type_as(flow)
    flow_grid = torch.clamp((flow + 1) * (H / 2), 0, H - 1)
    flow_grid = torch.cat((flow_grid[:, 1:2, :, :], flow_grid[:, 0:1, :, :]), 1)
    flow = flow_grid - base_grid

    image_numpy = flow.data[idx].cpu().float().numpy()
    image_numpy = flow2img(np.transpose(image_numpy, (1, 2, 0)))

    return image_numpy.astype(imtype)

def tensor2mask(input_image, idx=0, imtype=np.uint8):
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[idx].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

def tensor2att(input_image, idx=0, imtype=np.uint8):
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[idx].cpu().float().numpy()  # convert it into a numpy array
        image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    image_numpy = cv2.applyColorMap(image_numpy[:, :, 0].astype('uint8'), cv2.COLORMAP_JET)[:,:,::-1]
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    cv2.imwrite(image_path, cv2.cvtColor(image_numpy, cv2.COLOR_RGB2BGR))
    # image_pil = Image.fromarray(image_numpy)
    # image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))

def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
        self.deg = {'15':['050', '140'],
               '30':['041', '130'],
               '45':['080', '190'],
               '60':['090', '200'],
               '75':['010', '120'],
               '90':['110', '240']}

    def reset(self):
        self.stat_dict = {}

    def update(self, test_feas, test_names, gallery_feas, gallery_keys, topk=1):
        for b in range(test_feas.size(0)):
            name = basename(test_names[b])
            ss = name.split('_') # ss[0] is id and ss[3] is camera
            dis = torch.cosine_similarity(gallery_feas, test_feas[b:b + 1, :], dim=1)
            vvv, iii = dis.topk(k=max(10, topk), dim=0)
            iii = iii.cpu().data.numpy().squeeze().tolist()
            ids = [gallery_keys[ii] for ii in iii]
            if ss[3] not in self.stat_dict:
                self.stat_dict[ss[3]] = {'correct':0, 'all':0}
            self.stat_dict[ss[3]]['all'] += 1
            if ss[0] in ids[:topk]:
                self.stat_dict[ss[3]]['correct'] += 1

    def __str__(self):
        s, s1 = '', ''
        for k, v in self.stat_dict.items():
            s += '{}: [{}/{}, {}] \n'.format(k, v['correct'], v['all'], 1.0 * v['correct'] / v['all'])
        for k in self.deg:
            cameras = self.deg[k]
            _c, _a = 0, 0
            for c in cameras:
                _c += self.stat_dict[c]['correct']
                _a += self.stat_dict[c]['all']
            s += '{}: [{}/{}, {}] \n'.format(k, _c, _a, 1.0 * _c / _a)
            s1 += ' {:.2f} |'.format(100.0 * _c / _a)
        return s + s1 + '\n'