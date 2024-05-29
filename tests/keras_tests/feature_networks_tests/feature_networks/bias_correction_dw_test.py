# Copyright 2022 Sony Semiconductor Israel, Inc. All rights reserved.
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
import copy

import tensorflow as tf
import numpy as np
from keras.layers import DepthwiseConv2D
from keras.layers import Conv2D

from model_compression_toolkit.core.keras.constants import ACTIVATION, LINEAR
from tests.keras_tests.tpc_keras import get_quantization_disabled_keras_tpc
from tests.common_tests.helpers.tensors_compare import cosine_similarity
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest
import model_compression_toolkit as mct
from tests.keras_tests.feature_networks_tests.feature_networks.network_editor.node_filter_test import get_uniform_weights

from tests.keras_tests.utils import get_layers_from_model_by_type

keras = tf.keras
layers = keras.layers


class BiasCorrectionDepthwiseTest(BaseKerasFeatureNetworkTest):

    def __init__(self, unit_test):
        super().__init__(unit_test,
                         input_shape=(8,8,1))

    def get_quantization_config(self):
        return mct.core.QuantizationConfig(weights_bias_correction=True)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        outputs = DepthwiseConv2D(3, depth_multiplier=10, use_bias=False)(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model

    def generate_inputs(self):
        return np.ones(*self.get_input_shapes())

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        dw_layer = get_layers_from_model_by_type(quantized_model, Conv2D)[0]
        error = float_model.layers[1].depthwise_kernel - dw_layer.get_quantized_weights()['kernel']
        error = np.sum(error, axis=(0,1)).flatten()
        bias = dw_layer.weights[2]
        # Input mean is 1 so correction_term = quant_error * 1
        self.unit_test.assertTrue(np.isclose(error, bias.numpy(), atol=3e-7).all())
