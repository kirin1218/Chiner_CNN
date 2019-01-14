import os
import glob
import cv2
from PIL import Image
import numpy as np

import config as cf


class DataLoader():

    def __init__(self, phase='Train', shuffle=True):
        self.datas = []
        self.last_mb = 0
        self.phase = phase
        self.gt_count = [0 for _ in range(cf.Class_num)]
        self.prepare_datas(shuffle=shuffle)


    def prepare_datas(self, shuffle=True):

        if self.phase == 'Train':
            dir_paths = cf.Train_dirs
        elif self.phase == 'Test':
            dir_paths = cf.Test_dirs


        print()
        print('------------')
        print('Data Load (phase: {})'.format(self.phase))

        for dir_path in dir_paths:

            files = glob.glob(dir_path + '/*')

            load_count = 0
            for img_path in files:
                gt = self.get_gt(img_path)

                img = self.load_image(img_path)
                if img is not None:
                    gt_path = 1

                    data = {'img_path': img_path,
                            'gt_path': gt_path,
                            'h_flip': False,
                            'v_flip': False
                    }

                    self.datas.append(data)

                    self.gt_count[gt] += 1
                    load_count += 1

            print(' - {} - {} datas -> loaded {}'.format(dir_path, len(files), load_count))

        self.display_gt_statistic()

        if self.phase == 'Train':
            self.data_augmentation(h_flip=cf.Horizontal_flip, v_flip=cf.Vertical_flip)

            self.display_gt_statistic()


        self.set_index(shuffle=shuffle)



    def display_data_total(self):

        print('   Total data: {}'.format(len(self.datas)))


    def display_gt_statistic(self):

        print()
        print('  -*- Training label statistic -*-')
        self.display_data_total()

        for i, gt in enumerate(self.gt_count):
            print('   - {} : {}'.format(cf.Class_label[i], gt))



    def set_index(self, shuffle=True):
        self.data_n = len(self.datas)

        self.indices = np.arange(self.data_n)

        if shuffle:
            np.random.seed(0)
            np.random.shuffle(self.indices)



    def get_minibatch_index(self, shuffle=False):

        if self.phase == 'Train':
            mb = cf.Minibatch
        elif self.phase == 'Test':
            mb = cf.Minibatch

        _last = self.last_mb + mb

        if _last >= self.data_n:
            mb_inds = self.indices[self.last_mb:]
            self.last_mb = _last - self.data_n

            if shuffle:
                np.random.seed(0)
                np.random.shuffle(self.indices)

            _mb_inds = self.indices[:self.last_mb]
            mb_inds = np.hstack((mb_inds, _mb_inds))

        else:
            mb_inds = self.indices[self.last_mb : self.last_mb+mb]
            self.last_mb += mb

        self.mb_inds = mb_inds



    def get_minibatch(self, shuffle=True):

        if self.phase == 'Train':
            mb = cf.Minibatch
        elif self.phase == 'Test':
            mb = cf.Minibatch        

        self.get_minibatch_index(shuffle=shuffle)

        imgs = np.zeros((mb, 3, cf.Height, cf.Width), dtype=np.float32)

        # For Chainer
        gts = np.zeros((mb), dtype=np.int32)

        for i, ind in enumerate(self.mb_inds):
            data = self.datas[ind]
            img = self.load_image(data['img_path'], h_flip=data['h_flip'])
            gt = self.get_gt(data['img_path'])

            imgs[i] = img

            # For Chainer
            gts[i] = gt

        return imgs, gts



    def get_gt(self, img_name):

        for ind, cls in enumerate(cf.Class_label):
            if cls in img_name:
                return ind

        raise Exception("Class label Error {}".format(image_name))    

    def imgResize(self,img_path):
        # 縦横のサイズを指定
        width = cf.Width
        height = cf.Height

        # 画像の縦横を指定のサイズより小さくなるように変形
        img = Image.open(img_path)
        img.thumbnail((width,height),Image.ANTIALIAS)

        # 黒く塗りつぶす用の背景画像を作成
        bg = Image.new("RGBA",[width,height],(0,0,0,255))

        # 元の画像を、背景画像のセンターに配置
        bg.paste(img,(int((width-img.size[0])/2),int((height-img.size[1])/2)))
        ## Below functions are for data augmentation
        print(type(bg))

    def load_image(self, img_name, h_flip=False, v_flip=False):

        ## Image load
        self.imgResize(img_name)
        img = cv2.imread(img_name)

        if img is None:
            print('file not found: {}'.format(img_name))
            return img
            #raise Exception('file not found: {}'.format(img_name))
        img = cv2.resize(img, (cf.Width, cf.Height))
        print(type(img))
        #cv2.imshow('image', img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        img = cv2.resize(img, (cf.Width, cf.Height))
        img = img[:, :, (2,1,0)]
        img = img.transpose(2,0,1)
        img = img / 255.

        ## Horizontal flip
        if h_flip:
            img = img[:, ::-1, :]

        ## Vertical flip
        if v_flip:
            img = img[::-1, :, :]

        return img



    def data_augmentation(self, h_flip=False, v_flip=False):

        print()
        print('   ||   -*- Data Augmentation -*-')
        if h_flip:
            self.add_horizontal_flip()
            print('   ||    - Added horizontal flip')
        if v_flip:
            self.add_vertical_flip()
            print('   ||    - Added vertival flip')
        print('  \  /')
        print('   \/')



    def add_horizontal_flip(self):

        ## Add Horizontal flipped image data

        new_data = []

        for data in self.datas:
            _data = {'img_path': data['img_path'],
                     'gt_path': data['gt_path'],
                     'h_flip': True,
                     'v_flip': data['v_flip']
            }

            new_data.append(_data)

            gt = self.get_gt(data['img_path'])
            self.gt_count[gt] += 1

        self.datas.extend(new_data)



    def add_vertical_flip(self):

        ## Add Horizontal flipped image data

        new_data = []

        for data in self.datas:
            _data = {'img_path': data['img_path'],
                     'gt_path': data['gt_path'],
                     'h_flip': data['h_flip'],
                     'v_flip': True
            }

            new_data.append(_data)

            gt = self.get_gt(data['img_path'])
            self.gt_count[gt] += 1

        self.datas.extend(new_data)

