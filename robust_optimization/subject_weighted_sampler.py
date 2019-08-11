# -*- coding: utf-8 -*-
"""
Generating image window following a given discrete probability distribution
over the set of subjects and a uniformly probability distribution over the input image.

This can be considered as a generalization of the uniform sampler when the distribution over
the subjects is not necessarily uniform.
"""
from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

from niftynet.engine.image_window_dataset import ImageWindowDataset
from niftynet.engine.sampler_uniform_v2 import UniformSampler, rand_spatial_coordinates
from niftynet.engine.image_window import LOCATION_FORMAT


class SubjectWeightedSampler(UniformSampler):
    def __init__(self,
                 reader,
                 window_sizes,
                 batch_size=1,
                 windows_per_image=1,
                 queue_length=10,
                 name='subject_weighted_sampler'):

        ImageWindowDataset.__init__(
            self,
            reader=reader,
            window_sizes=window_sizes,
            batch_size=batch_size,
            windows_per_image=windows_per_image,
            queue_length=queue_length,
            shuffle=False,  # different from UniformSampler
            epoch=-1,
            smaller_final_batch_mode='drop',
            name=name)

        tf.logging.info("initialised subject weighted sampler %s ", self.window.shapes)
        tf.logging.warning("Should be used only during training!")
        self.window_centers_sampler = rand_spatial_coordinates

        # probability to draw each subject
        # in contains all subject so it is necessary to make sure that
        # the proba for validation and inference data are zero
        # initialised to uniform distribution
        self.subject_proba = [1./reader.num_subjects]*reader.num_subjects

    def set_subject_proba(self, new_proba):
        self.subject_proba = []
        # assign new proba and check that the keys correspond exactly to the one of the reader
        assert len(new_proba) == self.reader.num_subjects
        for idx in range(self.reader.num_subjects):
            pat_name = self.reader.get_subject(idx)['subject_id']
            self.subject_proba.append(new_proba[pat_name])

    # pylint: disable=too-many-locals
    def layer_op(self, idx=None):
        """
        This function generates sampling windows to the input buffer
        image data are from ``self.reader()``

        It starts by drawing the subject id according to self.subject_proba
        WARNING: as a result it will not return window samples for the subject idx
        given as input!

        Then it follows the same procedure as for the uniform sampler:
        it first completes window shapes based on image data,
        then finds random coordinates based on the window shapes
        finally extract window with the coordinates and output
        a dictionary (required by input buffer).

        :return: output data dictionary
            ``{image_modality: data_array, image_location: n_samples * 7}``
        """
        idx = np.random.choice(range(len(self.subject_proba)), 1, p=self.subject_proba)[0]
        # pat_name = self.reader.get_subject(idx)['subject_id']
        # print("Sample window from patient %s" % pat_name)
        image_id, data, _ = self.reader(idx=idx, shuffle=True)
        image_shapes = dict(
            (name, data[name].shape) for name in self.window.names)
        static_window_shapes = self.window.match_image_shapes(image_shapes)

        # find random coordinates based on window and image shapes
        coordinates = self._spatial_coordinates_generator(
            subject_id=image_id,
            data=data,
            img_sizes=image_shapes,
            win_sizes=static_window_shapes,
            n_samples=self.window.n_samples)

        # initialise output dict, placeholders as dictionary keys
        # this dictionary will be used in
        # enqueue operation in the form of: `feed_dict=output_dict`
        output_dict = {}
        # fill output dict with data
        for name in list(data):
            coordinates_key = LOCATION_FORMAT.format(name)
            image_data_key = name

            # fill the coordinates
            location_array = coordinates[name]
            output_dict[coordinates_key] = location_array

            # fill output window array
            image_array = []
            for window_id in range(self.window.n_samples):
                x_start, y_start, z_start, x_end, y_end, z_end = \
                    location_array[window_id, 1:]
                try:
                    image_window = data[name][
                        x_start:x_end, y_start:y_end, z_start:z_end, ...]
                    image_array.append(image_window[np.newaxis, ...])
                except ValueError:
                    tf.logging.fatal(
                        "dimensionality miss match in input volumes, "
                        "please specify spatial_window_size with a "
                        "3D tuple and make sure each element is "
                        "smaller than the image length in each dim. "
                        "Current coords %s", location_array[window_id])
                    raise
            if len(image_array) > 1:
                output_dict[image_data_key] = \
                    np.concatenate(image_array, axis=0)
            else:
                output_dict[image_data_key] = image_array[0]
        # the output image shape should be
        # [enqueue_batch_size, x, y, z, time, modality]
        # where enqueue_batch_size = windows_per_image
        return output_dict
