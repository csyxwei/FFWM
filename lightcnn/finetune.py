'''
    implement training process for Light CNN
    @author: Alfred Xiang Wu
    @date: 2017.07.04
'''
from __future__ import print_function
import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from os.path import basename, join
import numpy as np

from light_cnn import LightCNN_29Layers, LightCNN_29Layers_v2
from dataset import ImgDataset

parser = argparse.ArgumentParser(description='PyTorch Light CNN Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='LightCNN29')
parser.add_argument('--cuda', '-c', default=True)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--epochs', default=80, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=10, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=5000, type=int,
                    metavar='N', help='print frequency (default: 100)')
parser.add_argument('--model', default='LightCNN-29', type=str, metavar='Model',
                    help='model type: LightCNN-9, LightCNN-29, LightCNN-29v2')
parser.add_argument('--save_path', default='', type=str, metavar='PATH',
                    help='path to save checkpoint (default: none)')
parser.add_argument('--num_classes', default=79077, type=int,
                    metavar='N', help='number of classes (default: 99891)')
##############
parser.add_argument('--model_path', default='./', type=str)
parser.add_argument('--dataroot', default='../dataset', type=str)
parser.add_argument('--crop', action='store_true', help='center crop the face')
parser.add_argument('--preload', action='store_true', help='preload the img to memeory')

def main():
    global args
    args = parser.parse_args()

    # create Light CNN for face recognition
    if args.model == 'LightCNN-29':
        model = LightCNN_29Layers(num_classes=args.num_classes)
    elif args.model == 'LightCNN-29v2':
        model = LightCNN_29Layers_v2(num_classes=args.num_classes)
    else:
        print('Error model type\n')

    model = model.cuda()

    print(model)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # large lr for last fc parameters
    params = []
    for name, value in model.named_parameters():
        if 'bias' in name:
            if 'fc2' in name:
                params += [{'params': value, 'lr': 20 * args.lr, 'weight_decay': 0}]
            else:
                params += [{'params': value, 'lr': 2 * args.lr, 'weight_decay': 0}]
        else:
            if 'fc2' in name:
                params += [{'params': value, 'lr': 10 * args.lr}]
            else:
                params += [{'params': value, 'lr': 1 * args.lr}]

    optimizer = torch.optim.SGD(params, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    state_dict = torch.load(args.model_path)
    model.load_state_dict(state_dict)

    cudnn.benchmark = True
    # load image
    train_loader = torch.utils.data.DataLoader(
        ImgDataset(args.dataroot, False, args.crop, args.preload),
        batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        ImgDataset(args.dataroot, True, args.crop, args.preload),
        batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True)

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    criterion.cuda()

    validate(val_loader, model)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        validate(val_loader, model)

        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)

        if epoch % 5 == 0:
            save_checkpoint(model.state_dict(), join(args.save_path, 'lightCNN_' + str(epoch + 1) + '_checkpoint.pth'))
        save_checkpoint(model.state_dict(), join(args.save_path, 'lightCNN_latest_checkpoint.pth'))


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()

    end = time.time()
    for i, data in enumerate(train_loader):
        data_time.update(time.time() - end)

        input = data['img'].cuda()
        files = data['input_path']
        target = [int(x[:3]) - 1 for x in files]
        target = torch.LongTensor(target).cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        # compute output
        output, _, _ = model(input_var)
        loss = criterion(output, target_var)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
            f = open(join(args.save_path, 'logs.txt'), 'a+')
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5), file=f)
            f.close()


def validate(val_loader, model):
    top1 = MultiPIEAverageMeter()
    model.eval()
    gallery = val_loader.dataset.gallery_dict
    gallery_keys = list(gallery.keys())
    feas = []
    for key in gallery_keys:
        tensor = gallery[key].cuda()
        if len(tensor.size()) == 3:
            tensor = tensor.unsqueeze(0)
        if args.crop:
            tensor = tensor[:, :, 28:-2, 15:-15]
            tensor = torch.nn.functional.interpolate(tensor, (128, 128), mode='bilinear')
        _, f, _ = model(tensor)
        feas.append(f.detach())
    gallery_feas = torch.cat(feas, 0)

    for i, data in enumerate(val_loader):
        files = data['input_path']
        input = data['img'].cuda()
        _, ff, _ = model(input)
        top1.update(ff, files, gallery_feas, gallery_keys)

    f = open(join(args.save_path, 'logs.txt'), 'a+')
    print('\n Test Result: \n', top1)
    print('\n Test Result: \n', top1, file=f)
    f.close()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MultiPIEAverageMeter(object):
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

def save_checkpoint(state, filename):
    torch.save(state, filename)


def adjust_learning_rate(optimizer, epoch):
    scale = 0.457305051927326
    step = 25
    lr = args.lr * (scale ** (epoch // step))
    print('lr: {}'.format(lr))
    if (epoch != 0) & (epoch % step == 0):
        print('Change lr')
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * scale


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
