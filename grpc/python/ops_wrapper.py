# coding=utf-8
# Copyright 2019 The SEED Authors
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

"""Python wrapper around TensorFlow operations."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path


import tensorflow as tf

gen_grpc_ops = tf.load_op_library(
    os.path.join(tf.compat.v1.resource_loader.get_data_files_path(),
                 '../grpc_cc.so'))
import sys
print("***** grpc path", os.path.join(tf.compat.v1.resource_loader.get_data_files_path(),
                 '../grpc_cc.so'), file=sys.stderr)