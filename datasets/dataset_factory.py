# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# 2019-01-19
"""
A factory-pattern class which returns classification image/label pairs.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datasets import pascalvoc_2007

datasets_map = {
    'pascalvoc_2007': pascalvoc_2007,
}


def get_dataset(name, split_name, dataset_dir, file_pattern=None, reader=None):
    """
    Given a dataset name and split_name returns a Dataset.
    :param name:
    :param split_name:
    :param dataset_dir:
    :param file_pattern:
    :param reader:
    :return:
        A `Dataset` class.
    """
    if name not in datasets_map:
        raise ValueError('Name of dataset unknown %s.' % name)
    return datasets_map[name].get_split(split_name,
                                        dataset_dir,
                                        file_pattern,
                                        reader)