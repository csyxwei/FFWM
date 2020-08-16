import numpy as np
import os
import sys
import ntpath
import time
from . import util, html
from tensorboardX import SummaryWriter
from datetime import datetime

def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
    """Save images to the disk.

    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []

    for label, im_data in visuals.items():
        im = util.tensor2im(im_data)
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(image_dir, image_name)
        h, w, _ = im.shape
        if aspect_ratio > 1.0:
#            im = imresize(im, (h, int(w * aspect_ratio)), interp='bicubic')
            im = im.resize( (h, int(w * aspect_ratio)), interp='bicubic')

        if aspect_ratio < 1.0:
#            im = imresize(im, (int(h / aspect_ratio), w), interp='bicubic')
            im = im.resize( (int(h / aspect_ratio), w), interp='bicubic')

        util.save_image(im, save_path)

        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
    webpage.add_images(ims, txts, links, width=width)


class Visualizer():
    """This class includes several functions that can display/save images and print/save logging information.

    It uses a Python library 'visdom' for display, and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    def __init__(self, opt):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: connect to a visdom server
        Step 3: create an HTML object for saveing HTML filters
        Step 4: create a logging file to store training losses
        """
        self.opt = opt  # cache the option
        self.display_id = opt.display_id
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.port = opt.display_port
        self.saved = False

        now = datetime.now()

        self.test_dir = os.path.join(opt.checkpoints_dir, opt.name, 'test', opt.datamode)
        util.mkdirs([self.test_dir])

        if self.display_id > 0:
            self.save_dir = os.path.join(opt.checkpoints_dir, opt.name, 'log', now.strftime("%Y%m%d-%H%M"))
            self.writer = SummaryWriter(logdir=self.save_dir, flush_secs=1)
            util.mkdirs([self.save_dir])
            print('create log directory %s...' % self.save_dir)

        if self.use_html:  # create an HTML object at <checkpoints_dir>/web/; images will be saved under <checkpoints_dir>/web/images/
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])

        # create a logging file to store training losses
        self.loss_log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        self.test_log_name = os.path.join(opt.checkpoints_dir, opt.name, 'test_log.txt')
        with open(self.loss_log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    def display_current_results(self, visuals, epoch, save_result):
        """Display current results on visdom; save current results to an HTML file.

        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            epoch (int) - - the current epoch
            save_result (bool) - - if save the current results to an HTML file
        """
        if self.use_html and (save_result or not self.saved):  # save images to an HTML file if they haven't been saved.
            self.saved = True
            # save images to the disk
            for label, image in visuals.items():
                if 'mask' in label:
                    image_numpy = util.tensor2mask(image)
                else:
                    image_numpy = util.tensor2im(image)
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                util.save_image(image_numpy, img_path)
                if self.display_id > 0:
                    self.writer.add_image('%s/%s' % ('train', label),
                                          np.transpose(image_numpy, (2, 0, 1)).astype('uint8'),
                                          epoch)
                    self.writer.flush()
            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=1)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims, txts, links = [], [], []

                for label, image_numpy in visuals.items():
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    def display_test_results(self, visuals, epoch, save_result, name, idx=0):
        """Display current results on visdom; save current results to an HTML file.

        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            epoch (int) - - the current epoch
            save_result (bool) - - if save the current results to an HTML file
        """
        if save_result or not self.saved:  # save images to an HTML file if they haven't been saved.
            self.saved = True
            # save images to the disk
            for label, image in visuals.items():
                if 'att' in label:
                    image_numpy = util.tensor2att(image, idx=idx)
                elif 'mask' in label:
                    image_numpy = util.tensor2mask(image, idx=idx)
                elif 'flow' in label:
                    image_numpy = util.tensor2flow(image, idx=idx)
                else:
                    image_numpy = util.tensor2im(image, idx=idx)
                img_path = os.path.join(self.test_dir, '%s_%s.png' % (name, label))
                util.save_image(image_numpy, img_path)
                if self.display_id > 0:
                    self.writer.add_image('%s/%s_%s' % ('test', name, label),
                                          np.transpose(image_numpy, (2, 0, 1)).astype('uint8'),
                                          epoch)
                    self.writer.flush()

    def print_test_results(self, topk):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = str(topk)
        print(message)  # print the message
        with open(self.test_log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message

    def print_current_losses(self, epoch, iters, losses, t_comp, t_data, total_steps):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.7f ' % (k, v)
            if self.display_id > 0:
                if total_steps == 0:
                    self.writer.add_scalar('epoch_loss/%s' % k, v, epoch)
                else:
                    self.writer.add_scalar('iter_loss/%s' % k, v, total_steps)
                self.writer.flush()
        print(message)  # print the message
        with open(self.loss_log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message

