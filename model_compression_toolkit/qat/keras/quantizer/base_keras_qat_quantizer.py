# Copyright 2023 Sony Semiconductor Israel, Inc. All rights reserved.
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
from typing import Union

from model_compression_toolkit.logger import Logger
from model_compression_toolkit.constants import FOUND_TF

from model_compression_toolkit.trainable_infrastructure import TrainableQuantizerWeightsConfig, \
    TrainableQuantizerActivationConfig, BaseKerasTrainableQuantizer

if FOUND_TF:

    class BaseKerasQATTrainableQuantizer(BaseKerasTrainableQuantizer):
        """
        A base class for trainable Keras quantizer for QAT.
        """

        def __init__(self,
                     quantization_config: Union[TrainableQuantizerWeightsConfig, TrainableQuantizerActivationConfig]):
            """
            Initializes BaseKerasQATTrainableQuantizer object.

            Args:
                quantization_config: quantizer config class contains all the information about a quantizer configuration.
            """

            super().__init__(quantization_config)

else:  # pragma: no cover
    class BaseKerasQATTrainableQuantizer(BaseKerasTrainableQuantizer):
        def __init__(self,
                     quantization_config: Union[TrainableQuantizerWeightsConfig, TrainableQuantizerActivationConfig]):

            super().__init__(quantization_config)
            Logger.critical("Tensorflow must be installed to use BaseKerasQATTrainableQuantizer. "
                            "The 'tensorflow' package is missing.")  # pragma: no cover
