import os
import numpy as np
import tensorflow as tf
from config import Config
from ops import tf_fun


class data_processing(object):
    def __init__(self):
        self.name = 'gratings_undo_bias_control'
        self.output_name = 'gratings_undo_bias_control'
        self.img_dir = 'imgs'
        self.contour_dir = '/media/data_cifs/tilt_illusion'
        self.im_extension = '.png'
        self.label_regex = r'(?<=length)\d+'
        self.config = Config()
        self.im_size = [500, 500]  # 600, 600
        self.model_input_image_size = [224, 224, 1]  # [107, 160, 3]
        self.output_size = [2]
        self.max_ims = 120
        self.label_size = self.output_size
        self.default_loss_function = 'l2'
        self.store_z = False
        self.normalize_im = False
        self.all_flips = True
        self.shuffle = True
        self.input_normalization = 'none'  # 'zscore'
        self.preprocess = []  # ['resize']  # ['resize_nn']
        self.meta = os.path.join('train', 'metadata', 'filtered_tr.npy')
        self.folds = {
            'train': 'train',
            'val': 'val',
            'test': 'test',
        }
        self.cv_split = 0.1
        self.targets = {
            'image': tf_fun.bytes_feature,
            'label': tf_fun.float_feature
        }
        self.tf_dict = {
            'image': tf_fun.fixed_len_feature(dtype='string'),
            'label': tf_fun.fixed_len_feature(
                dtype='float32',
                length=self.output_size)
        }
        self.tf_reader = {
            'image': {
                'dtype': tf.float32,
                'reshape': self.im_size
            },
            'label': {
                'dtype': tf.float32,
                'reshape': self.output_size
            }
        }

    def list_files(self, meta, directory):
        """List files from metadata."""
        files, labs = [], []
        for idx, f in enumerate(meta):
            files += [
                os.path.join(
                    self.contour_dir,
                    directory,
                    f[0],
                    f[1])]
            deg = float(f[4])
            labs += [[np.sin(deg * np.pi / 180.), np.cos(deg * np.pi / 180.)]]
        return np.asarray(files), np.asarray(labs)

    def get_data(self):
        """Get the names of files."""
        meta_train = np.load(
            os.path.join(
                self.contour_dir,
                self.meta))
        meta_train = meta_train.reshape(-1, 11)
        all_ims, all_labs = self.list_files(
            meta_train, 'train')
        np.random.seed(1)
        if self.max_ims:
            rand_idx = np.random.permutation(len(all_ims))
            all_ims = all_ims[rand_idx]
            all_labs = all_labs[rand_idx]
            all_ims = all_ims[:self.max_ims]
            all_labs = all_labs[:self.max_ims]
        num_ims = len(all_ims)

        # Balance CV folds
        val_cutoff = np.round(num_ims * (1 - self.cv_split)).astype(int)
        shuffle_idx = np.random.permutation(num_ims)
        shuffle_ims = all_ims[shuffle_idx]
        shuffle_labs = all_labs[shuffle_idx]
        train_ims = shuffle_ims[:val_cutoff]
        train_labs = shuffle_labs[:val_cutoff]
        val_ims = shuffle_ims[val_cutoff:]
        val_labs = shuffle_labs[val_cutoff:]

        # Build CV dict
        cv_files, cv_labels = {}, {}
        cv_files[self.folds['train']] = train_ims
        cv_files[self.folds['val']] = val_ims
        cv_files[self.folds['test']] = train_ims
        cv_labels[self.folds['train']] = train_labs
        cv_labels[self.folds['val']] = val_labs
        cv_labels[self.folds['test']] = train_labs
        return cv_files, cv_labels
