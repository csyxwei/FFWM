import torch
import torch.utils.data as data
from os.path import basename, join
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import cv2

class ImgDataset(data.Dataset):
    def __init__(self, dataroot, isval=False, crop=False, preload=True):
        self.isval = isval
        self.dataroot = dataroot
        self.crop = crop
        self.preload = preload
        self.image_dict = {}  # for preload
        self.img_list = self.get_list()
        self.load_size = 128
        self.transforms = transforms.Compose([transforms.ToPILImage(mode=None),
                                              transforms.RandomRotation(5, resample=Image.BICUBIC, expand=False,
                                                                        center=None),
                                              transforms.RandomHorizontalFlip(p=0.5),
                                              transforms.ToTensor(),
                                              ])


    def __getitem__(self, index):
        if self.isval:
            return self.get_test_item(index)
        else:
            return self.get_train_item(index)

    def get_test_item(self, index):
        path = self.img_list[index]
        img = self.image_transform(path, self.preload)
        img = torch.from_numpy(img.transpose((2, 0, 1)).astype('float32'))
        img = self.postprocess(img, train=False)
        return {'img': img, 'input_path': path}

    def get_train_item(self, index):
        path= self.img_list[index]
        img = self.image_transform(path, self.preload)
        img = torch.from_numpy(img.transpose((2, 0, 1)).astype('float32'))
        img = self.postprocess(img, train=True)
        return {'img':img, 'input_path': path}


    def image_transform(self, file, preload=False):
        if preload:
            return self.image_dict[file].copy().astype('float32')
        else:
            image_path = join(self.base_path, 'images', file)
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img.astype('uint8')

    def postprocess(self, img, train=False):
        if train:
            img = self.transforms(img.byte())
        else:
            img = img.float().div(255)
        img = torch.mean(img, dim=(0, ), keepdim=True)
        if self.crop:
            img = img[:, 28:-2, 15:-15]
            img = torch.nn.functional.interpolate(img.unsqueeze(0), (self.load_size, self.load_size), mode='bilinear')
            img = img[0]
        return img

    def get_list(self):
        if self.isval:
            self.base_path = join(self.dataroot, 'test')
            self.files = os.listdir(join(self.base_path, 'images'))
            self.gallery_dict = self.get_gallery()
        else:
            self.base_path = join(self.dataroot, 'train')
            self.files = os.listdir(join(self.base_path, 'images'))
        if self.preload:
            read_images(self)
        return self.files

    def get_gallery(self):
        if os.path.exists(join(self.base_path, 'gallery_list.npy')):
            gallery_list = np.load(join(self.base_path, 'gallery_list.npy'))
        else:
            _dict = {}
            np.random.shuffle(self.files)
            for k in self.files:
                if k[:3] not in _dict and k.strip().endswith('051_06.png'):
                    _dict[k[:3]] = k
            gallery_list = _dict.values()
        gallery_dict = {}
        for g in gallery_list:
            gallery = self.image_transform(g)
            gallery = torch.from_numpy(gallery.transpose((2, 0, 1)).astype('float32')).div(255)
            gallery_dict[g[:3]] = torch.mean(gallery, (0, ), keepdim=True)
        return gallery_dict

    def __len__(self):
        return len(self.img_list)


def iter_obj(num, objs):
    for i in range(num):
        yield (i, objs)

def imreader(arg):
    i, obj = arg
    for _ in range(3):
        try:
            obj.image_dict[obj.files[i]] = obj.image_transform(obj.files[i])
            failed = False
            break
        except Exception as e:
            print(e)
            failed = True
    if failed: print('%s fails!' % obj.files[i])

def read_images(obj):
    # can change to `from multiprocessing import Pool`, but less efficient and
    # NOTE: `multiprocessing.Pool` will duplicate given object for each process
    # therefore using `multiprocessing.dummy.Pool` is more convenient/efficient
    from multiprocessing.dummy import Pool
    from tqdm import tqdm
    print('Starting to load images via multiple imreaders')
    pool = Pool() # use all threads by default
    for _ in tqdm(pool.imap(imreader, iter_obj(len(obj.files), obj)), total=len(obj.files)):
        pass
    pool.close()
    pool.join()