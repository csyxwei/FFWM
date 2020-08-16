import os, cv2, torch
import numpy as np
from os.path import basename, join
from data.base_dataset import BaseDataset

"""
Face Dataset
"""

def s2f(file):
    """
    get corresponding frontal image name: only for multipie
    """
    path, name = os.path.split(file)
    ss = name.split('_')
    name = '{}_{}_{}_{}_{}'.format(ss[0], ss[1], ss[2], '051', ss[4])
    return name


class FaceDataset(BaseDataset):
    def __init__(self, opt, isval=False):
        BaseDataset.__init__(self, opt)
        self.preload = opt.preload
        self.load_size = opt.load_size
        self.opt = opt
        self.isval = isval # train or test dataset
        self.image_dict = {}  # for preload
        self.mask_dict = {} # for preload
        self.pairs = self.get_pairs()

    def __getitem__(self, index):
        if self.isval:
            return self.get_test_item(index)
        else:
            return self.get_train_item(index)

    def get_test_item(self, index):
        path_S, path_F = self.pairs[index]
        img_S = self.image_transform(path_S, preload=self.preload)
        img_F = self.image_transform(path_F, preload=self.preload)
        img_S = torch.from_numpy(img_S.transpose((2, 0, 1)).astype('float32')).div(255)
        img_F = torch.from_numpy(img_F.transpose((2, 0, 1)).astype('float32')).div(255)
        return {'img_S': img_S, 'img_F': img_F, 'input_path': path_S}

    def get_train_item(self, index):
        # Flip Augment
        if index >= len(self.pairs):
            _index = index % len(self.pairs)
        else:
            _index = index

        path_S, path_F = self.pairs[_index]
        key_S, key_F = path_S[:-7], path_F[:-7]

        lm_S = self.lm_dicts['lm_S'][key_S].copy()
        lm_F = self.lm_dicts['lm_F'][key_F].copy()
        gate = self.lm_dicts['gate'][key_S].copy()

        img_S = self.image_transform(path_S, preload=self.preload)
        img_F = self.image_transform(path_F, preload=self.preload)
        mask_S = self.mask_transform(path_S, preload=self.preload)
        mask_F = self.mask_transform(path_F, preload=self.preload)

        # Flip image, mask, and landmark
        if index >= len(self.pairs):
            lm_S = np.hstack((127 - lm_S[:, 0:1], lm_S[:, 1:2]))
            lm_F = np.hstack((127 - lm_F[:, 0:1], lm_F[:, 1:2]))
            img_S = img_S[:, ::-1, :]
            img_F = img_F[:, ::-1, :]
            mask_S = mask_S[:, ::-1, :]
            mask_F = mask_F[:, ::-1, :]

        # random rotation
        if self.opt.aug:
            img_S, mask_S, lm_S = self.aug_transform(img_S, mask_S, lm_S)

        img_S = torch.from_numpy(img_S.transpose((2, 0, 1)).astype('float32')).div(255)
        img_F = torch.from_numpy(img_F.transpose((2, 0, 1)).astype('float32')).div(255)
        mask_S = torch.from_numpy(mask_S.transpose((2, 0, 1)).astype('float32')).div(255)
        mask_F = torch.from_numpy(mask_F.transpose((2, 0, 1)).astype('float32')).div(255)

        lm_S = torch.from_numpy(lm_S).long()
        lm_S = torch.clamp(lm_S, 0, self.load_size - 1)
        lm_F = torch.from_numpy(lm_F).long()
        lm_F = torch.clamp(lm_F, 0, self.load_size - 1)
        gate = torch.from_numpy(gate.astype('float32')).unsqueeze(1)

        return {'img_S': img_S, 'img_F': img_F, 'input_path': path_S,
                'lm_S': lm_S, 'lm_F': lm_F, 'gate': gate,
                'mask_S': mask_S, 'mask_F': mask_F}

    def image_transform(self, file, preload=False):
        if preload:
            return self.image_dict[file].copy().astype('float32')
        else:
            image_path = join(self.base_path, 'images', file)
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img.astype('uint8')

    def mask_transform(self, file, preload=False):
        if preload:
            return self.mask_dict[file].copy().astype('float32')
        else:
            mask_path = join(self.base_path, 'masks', file)
            mask = cv2.imread(mask_path, 0)
            mask = mask[:, :, np.newaxis]
            return mask.astype('uint8')

    def aug_transform(self, img, mask, lm):
        ang = np.random.randint(-5, 5)
        ### rotation
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        mat = cv2.getRotationMatrix2D(center, int(ang), 1)
        img_aug = cv2.warpAffine(img, mat, (w, h))
        mask_aug = cv2.warpAffine(mask, mat, (w, h))
        mask_aug[mask_aug > 0] = 255
        mask_aug = mask_aug[:, :, np.newaxis]

        ### landmark
        lm_aug = lm.astype('float32')
        x0 = lm_aug[:, 0] - (self.load_size // 2)
        y0 = lm_aug[:, 1] - (self.load_size // 2)
        # Note: the angle shoud be -ang to ensure it is consistent with opencv
        ang_arc = -ang * np.pi / 180.0
        lm_aug[:, 0] = x0 * np.cos(ang_arc) - y0 * np.sin(ang_arc) + (self.load_size // 2)
        lm_aug[:, 1] = x0 * np.sin(ang_arc) + y0 * np.cos(ang_arc) + (self.load_size // 2)
        lm_aug = np.clip(lm_aug, 0, self.load_size)
        return img_aug, mask_aug, lm_aug

    def get_pairs(self):
        dataroot = join(self.opt.dataroot, self.opt.datamode)
        if self.opt.datamode == 'multipie':
            if self.isval:
                self.base_path = join(dataroot, 'test')
                self.files = os.listdir(join(self.base_path, 'images'))
                self.gallery_dict = self.get_gallery()
            else:
                self.base_path = join(dataroot, 'train')
                self.lm_dicts = np.load(join(self.base_path, 'landmarks.npy'), allow_pickle=True).item()
                self.files = os.listdir(join(self.base_path, 'images'))
            pairs = [(file, s2f(file)) for file in self.files]
        else:  # LFW or others
            self.base_path = dataroot
            self.files = os.listdir(join(self.base_path, 'images'))
            pairs = [(file, file) for file in self.files]  # no frontal file

        if self.preload: # preload images and masks to memory
            read_images(self)
        return pairs

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
        if self.isval:
            return len(self.pairs)
        else:
            return len(self.pairs) * 2


#### multiprocessing

def iter_obj(num, objs):
    for i in range(num):
        yield (i, objs)

def imreader(arg):
    i, obj = arg
    for _ in range(3):
        try:
            obj.image_dict[obj.files[i]] = obj.image_transform(obj.files[i])
            if not obj.isval:
                obj.mask_dict[obj.files[i]] = obj.mask_transform(obj.files[i])
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