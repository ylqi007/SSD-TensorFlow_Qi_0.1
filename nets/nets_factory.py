# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# 20190120
"""
Contains a factory for building various models.
"""

import functools
import tensorflow as tf

from nets import ssd_vgg_300

slim = tf.contrib.slim

networks_map = {'ssd_300_vgg': ssd_vgg_300.ssd_net,}

arg_scopes_map = {'ssd_300_vgg': ssd_vgg_300.ssd_arg_scope,}

networks_obj = {'ssd_300_vgg': ssd_vgg_300.SSDNet,}


def get_network(name):
    """
    Get a network object from a name.

    :return:
        A network class.
    """
    return networks_obj[name]


def get_network_fn(name, num_classes, is_training=False, **kwargs):
    """
    :param name: The name of the network.
    :param num_classes: The number of classes to use for classification.
    :param is_training: `True` if the model is being used for training and `False` otherwise.
    :param kwargs:
    :return:
        network_fn: A function that applies the model to a batch of images. It has the following signature:
            logits, end_points = network_fn(images)
    :raise
        ValueError: If network `name` is not recognized.
    """
    if name not in networks_map:
        raise ValueError('Name of network unknown %s' % name)
    arg_scope = arg_scopes_map[name](**kwargs)
    func = networks_map[name]
    @functools.wraps(func)
    def network_fn(images, **kwargs):
        with slim.arg_scope(arg_scope):
            return func(images, num_classes, is_training=is_training, **kwargs)
    if hasattr(func, 'default_image_size'):
        network_fn.default_image_size = func.default_image_size

    return network_fn
