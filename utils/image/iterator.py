"""Utilities for real-time data augmentation on image data.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading
import numpy as np
from keras_preprocessing import get_keras_submodule

try:
    IteratorType = get_keras_submodule('utils').Sequence
except ImportError:
    IteratorType = object

from .utils import (array_to_img,
                    img_to_array,
                    load_img)
from .affine_transformations import random_crop, center_crop, rotate_random_zoom_crop


class Iterator(IteratorType):
    """Base class for image data iterators.

    Every `Iterator` must implement the `_get_batches_of_transformed_samples`
    method.

    # Arguments
        n: Integer, total number of samples in the dataset to loop over.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seeding for data shuffling.
    """
    white_list_formats = {'png', 'jpg', 'jpeg', 'bmp', 'ppm', 'tif', 'tiff'}

    def __init__(self, n, batch_size, shuffle, seed):
        self.n = n
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_array = None
        self.index_generator = self._flow_index()

    def _set_index_array(self):
        self.index_array = np.arange(self.n)
        if self.shuffle:
            self.index_array = np.random.permutation(self.n)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise ValueError('Asked to retrieve element {idx}, '
                             'but the Sequence '
                             'has length {length}'.format(idx=idx,
                                                          length=len(self)))
        if self.seed is not None:
            np.random.seed(self.seed + self.total_batches_seen)
        self.total_batches_seen += 1
        if self.index_array is None:
            self._set_index_array()
        index_array = self.index_array[self.batch_size * idx:
                                       self.batch_size * (idx + 1)]
        return self._get_batches_of_transformed_samples(index_array)

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size  # round up

    def on_epoch_end(self):
        self._set_index_array()

    def reset(self):
        self.batch_index = 0

    def _flow_index(self):
        # Ensure self.batch_index is 0.
        self.reset()
        while 1:
            if self.seed is not None:
                np.random.seed(self.seed + self.total_batches_seen)
            if self.batch_index == 0:
                self._set_index_array()

            current_index = (self.batch_index * self.batch_size) % self.n
            if self.n > current_index + self.batch_size:
                self.batch_index += 1
            else:
                self.batch_index = 0
            self.total_batches_seen += 1
            yield self.index_array[current_index:
                                   current_index + self.batch_size]

    def __iter__(self):
        # Needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)

    def _get_batches_of_transformed_samples(self, index_array):
        """Gets a batch of transformed samples.

        # Arguments
            index_array: Array of sample indices to include in batch.

        # Returns
            A batch of transformed samples.
        """
        raise NotImplementedError


class BatchFromFilesMixin():
    """Adds methods related to getting batches from filenames

    It includes the logic to transform image files to batches.
    """

    def set_processing_attrs(self,                             
                             image_data_generator,
                             target_size,
                             crop_size,
                             color_mode,
                             data_format,
                             save_to_dir,
                             save_prefix,
                             save_format,
                             subset,
                             interpolation):
        """Sets attributes to use later for processing files into a batch.

        # Arguments
            image_data_generator: Instance of `ImageDataGenerator`
                to use for random transformations and normalization.
            target_size: tuple of integers, dimensions to resize input images to.
            color_mode: One of `"rgb"`, `"rgba"`, `"grayscale"`.
                Color mode to read images.
            data_format: String, one of `channels_first`, `channels_last`.
            save_to_dir: Optional directory where to save the pictures
                being yielded, in a viewable format. This is useful
                for visualizing the random transformations being
                applied, for debugging purposes.
            save_prefix: String prefix to use for saving sample
                images (if `save_to_dir` is set).
            save_format: Format to use for saving sample images
                (if `save_to_dir` is set).
            subset: Subset of data (`"training"` or `"validation"`) if
                validation_split is set in ImageDataGenerator.
            interpolation: Interpolation method used to resample the image if the
                target size is different from that of the loaded image.
                Supported methods are "nearest", "bilinear", and "bicubic".
                If PIL version 1.1.3 or newer is installed, "lanczos" is also
                supported. If PIL version 3.4.0 or newer is installed, "box" and
                "hamming" are also supported. By default, "nearest" is used.
        """
        if crop_size is not None:
            self.crop_size = tuple(crop_size)
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'rgba', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb", "rgba", or "grayscale".')
        self.color_mode = color_mode
        self.data_format = data_format
        if self.color_mode == 'rgba':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (4,)
                if crop_size is not None:
                    self.crop_shape = self.crop_size + (4,)
                
            else:
                self.image_shape = (4,) + self.target_size
                if crop_size is not None:
                    self.crop_shape = (4,) + self.crop_size 
        elif self.color_mode == 'rgb':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (3,)
                if crop_size is not None:
                    self.crop_shape = self.crop_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
                if crop_size is not None:
                    self.crop_shape = (3,) + self.crop_size 
        else:
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (1,)
                if crop_size is not None:
                    self.crop_shape = self.crop_size + (1,)
                
            else:
                self.image_shape = (1,) + self.target_size
                if crop_size is not None:
                    self.crop_shape = (1,) + self.crop_size 
                    
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.interpolation = interpolation
        if subset is not None:
            validation_split = self.image_data_generator._validation_split
            if subset == 'validation':
                split = (0, validation_split)
            elif subset == 'training':
                split = (validation_split, 1)
            else:
                raise ValueError(
                    'Invalid subset name: %s;'
                    'expected "training" or "validation"' % (subset,))
        else:
            split = None
        self.split = split
        self.subset = subset

    def _get_batches_of_transformed_samples(self, index_array):
        """Gets a batch of transformed samples.

        # Arguments
            index_array: Array of sample indices to include in batch.

        # Returns
            A batch of transformed samples.
        """
        
        #if (self.image_data_generator.return_add_img is not False) and (self.image_data_generator.n_imgs ==1):
            #print('Please set the attribute "n_img" to a number higher than 1')
            #break
            
        if (self.image_data_generator.return_add_img is not False):
            if (self.image_data_generator.random_crop is not False) or (self.image_data_generator.center_crop is not False) or (self.image_data_generator.rotate_random_zoom_crop is not False):
            #print(self.image_data_generator.center_crop)
            #print(self.image_data_generator.random_crop )
            #print(self.image_data_generator.rotate_random_zoom_crop)
            #k=0
           # for i in range(self.image_data_generator.n_imgs):
                batch_xc = np.zeros((len(index_array),) + self.crop_shape, dtype=self.dtype)
                  #  k+=1
                #print(batch_x.shape)
                batch_xo = np.zeros((len(index_array),) + self.image_shape, dtype=self.dtype)
                # build batch of image data
                # self.filepaths is dynamic, is better to call it once outside the loop
                filepaths = self.filepaths
                for i, j in enumerate(index_array):

                    img = load_img(filepaths[j],
                                       color_mode=self.color_mode,
                                       target_size=self.target_size,
                                       interpolation=self.interpolation)
                    x = img_to_array(img, data_format=self.data_format)
                    o = x

                    if self.image_data_generator.random_crop:     
                         x = random_crop(x, crop_width=self.crop_size[1], crop_height=self.crop_size[0])

                    elif self.image_data_generator.center_crop:                          
                         x = center_crop(x, crop_width=self.crop_size[1], crop_height=self.crop_size[0])

                    elif self.image_data_generator.rotate_random_zoom_crop:                             
                         x = rotate_random_zoom_crop(x, crop_width=self.crop_size[1], crop_height=self.crop_size[0])           

                        # Pillow images should be closed after `load_img`,
                        # but not PIL images.
                    if hasattr(img, 'close'):
                        img.close()
                    if self.image_data_generator:
                        params = self.image_data_generator.get_random_transform(x.shape)
                        x = self.image_data_generator.apply_transform(x, params)
                        x = self.image_data_generator.standardize(x)
                        paramso = self.image_data_generator.get_random_transform(o.shape)
                        o = self.image_data_generator.apply_transform(o, paramso)
                        o = self.image_data_generator.standardize(o)
                    batch_xc[i] = x
                    batch_xo[i] = o



                # optionally save augmented images to disk for debugging purposes
                if self.save_to_dir:
                    for i, j in enumerate(index_array):
                        img = array_to_img(batch_xc[i], self.data_format, scale=True)
                        imgo = array_to_img(batch_xo[i], self.data_format, scale=True)
                        fname = '{prefix}_{index}_{hash}.{format}'.format(
                            prefix=self.save_prefix,
                            index=j,
                            hash=np.random.randint(1e7),
                            format=self.save_format)
                        img.save(os.path.join(self.save_to_dir, fname))
                        imgo.save(os.path.join(self.save_to_dir, fname))
                # build batch of labels
                if self.class_mode == 'input':
                    batch_yc = batch_xc.copy()
                    batch_yo = batch_xo.copy()
                elif self.class_mode in {'binary', 'sparse'}:
                    batch_yc = np.empty(len(batch_xc), dtype=self.dtype)
                    batch_yo = np.empty(len(batch_xo), dtype=self.dtype)
                    for i, n_observation in enumerate(index_array):
                        batch_yc[i] = self.classes[n_observation]
                        batch_yo[i] = self.classes[n_observation]
                elif self.class_mode == 'categorical':
                    batch_yc = np.zeros((len(batch_xc), len(self.class_indices)),
                                       dtype=self.dtype)
                    batch_yo = np.zeros((len(batch_xo), len(self.class_indices)),
                                       dtype=self.dtype)
                    for i, n_observation in enumerate(index_array):
                        batch_yc[i, self.classes[n_observation]] = 1.
                        batch_yo[i, self.classes[n_observation]] = 1.
                elif self.class_mode == 'other':
                    batch_yc = self.data[index_array]
                    batch_yo = self.data[index_array]
                else:
                    return [batch_xo, batch_xc]
                return [batch_xo, batch_xc], batch_yo     
            else:
                print('You need to specify a cropping method:\n random_crop, center_crop or rotate_random_zoom_crop')


            
            
            
        elif (self.image_data_generator.return_add_img is False):
        
            if (self.image_data_generator.random_crop is not False) or (self.image_data_generator.center_crop is not False) or (self.image_data_generator.rotate_random_zoom_crop is not False):
                #print(self.image_data_generator.center_crop)
                #print(self.image_data_generator.random_crop )
                #print(self.image_data_generator.rotate_random_zoom_crop)
                batch_x = np.zeros((len(index_array),) + self.crop_shape, dtype=self.dtype)
                #print(batch_x.shape)
            else:
                batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=self.dtype)
            # build batch of image data
            # self.filepaths is dynamic, is better to call it once outside the loop
            filepaths = self.filepaths
            for i, j in enumerate(index_array):

                img = load_img(filepaths[j],
                                   color_mode=self.color_mode,
                                   target_size=self.target_size,
                                   interpolation=self.interpolation)
                x = img_to_array(img, data_format=self.data_format)


                if self.image_data_generator.random_crop:     
                     x = random_crop(x, crop_width=self.crop_size[1], crop_height=self.crop_size[0])

                elif self.image_data_generator.center_crop:                          
                     x = center_crop(x, crop_width=self.crop_size[1], crop_height=self.crop_size[0])

                elif self.image_data_generator.rotate_random_zoom_crop:                             
                     x = rotate_random_zoom_crop(x, crop_width=self.crop_size[1], crop_height=self.crop_size[0])           

                    # Pillow images should be closed after `load_img`,
                    # but not PIL images.
                if hasattr(img, 'close'):
                    img.close()
                if self.image_data_generator:
                    params = self.image_data_generator.get_random_transform(x.shape)
                    x = self.image_data_generator.apply_transform(x, params)
                    x = self.image_data_generator.standardize(x)
                batch_x[i] = x

            # optionally save augmented images to disk for debugging purposes
            if self.save_to_dir:
                for i, j in enumerate(index_array):
                    img = array_to_img(batch_x[i], self.data_format, scale=True)
                    fname = '{prefix}_{index}_{hash}.{format}'.format(
                        prefix=self.save_prefix,
                        index=j,
                        hash=np.random.randint(1e7),
                        format=self.save_format)
                    img.save(os.path.join(self.save_to_dir, fname))
            # build batch of labels
            if self.class_mode == 'input':
                batch_y = batch_x.copy()
            elif self.class_mode in {'binary', 'sparse'}:
                batch_y = np.empty(len(batch_x), dtype=self.dtype)
                for i, n_observation in enumerate(index_array):
                    batch_y[i] = self.classes[n_observation]
            elif self.class_mode == 'categorical':
                batch_y = np.zeros((len(batch_x), len(self.class_indices)),
                                   dtype=self.dtype)
                for i, n_observation in enumerate(index_array):
                    batch_y[i, self.classes[n_observation]] = 1.
            elif self.class_mode == 'other':
                batch_y = self.data[index_array]
            else:
                return batch_x
            return batch_x, batch_y

    @property
    def filepaths(self):
        """List of absolute paths to image files"""
        raise NotImplementedError(
            '`filepaths` property method has not been implemented in {}.'
            .format(type(self).__name__)
        )

    @property
    def labels(self):
        """Class labels of every observation"""
        raise NotImplementedError(
            '`labels` property method has not been implemented in {}.'
            .format(type(self).__name__)
        )

    @property
    def data(self):
        raise NotImplementedError(
            '`data` property method has not been implemented in {}.'
            .format(type(self).__name__)
        )
