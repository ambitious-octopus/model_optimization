# QAT Quantizers

## Introduction
[`BaseKerasQATTrainableQuantizer`](./quantizer/base_keras_qat_quantizer.py) is an interface that utilizes the Quantization Infrastructure's [`BaseKerasTrainableQuantizer`](../../quantizers_infrastructure/keras/base_keras_quantizer.py) class to  enable easy development of quantizers dedicated to Quantization-Aware Training (QAT).
All available training types for QAT are defined in the Enum `TrainingMethod`. See [Training Methods for QAT](./quantizer/README.md) for more details.

## Make your own Keras trainable quantizers
A trainable quantizer can be Weights Quantizer or Activation Quantizer.
In order to make your new quantizer you need to create your quantizer class, `MyTrainingQuantizer` and do as follows:
   - `MyTrainingQuantizer` should inherit from `BaseKerasQATTrainableQuantizer`.
   - `MyTrainingQuantizer` should have [`init`](../../quantizers_infrastructure/common/base_trainable_quantizer.py) function that gets `quantization_config` which is [`NodeWeightsQuantizationConfig`](../../core/common/quantization/node_quantization_config.py#L228) if you choose to implement weights quantizer or [`NodeActivationQuantizationConfig`](../../core/common/quantization/node_quantization_config.py#L63) if you choose activation quantizer.
   - Implement [`initialize_quantization`](../../quantizers_infrastructure/common/base_trainable_quantizer.py) where you can define your parameters for the quantizer.
   - Implement [`__call__`](../../quantizers_infrastructure/common/base_trainable_quantizer.py) method to quantize the given inputs while training. This is your custom quantization itself. 
   - Implement [`convert2inferable`](../../quantizers_infrastructure/common/base_trainable_quantizer.py) method. This method exports your quantizer for inference (deployment). For doing that you need to choose one of our Inferable Quantizers ([Inferable Quantizers](../../quantizers_infrastructure/keras/inferable_quantizers)) according to target when implementing `convert2inferable`, and set your learned quantization parameters there.
   - Decorate `MyTrainingQuantizer` class with the `@mark_quantizer` decorator and choose the appropriate properties to set for you quantizer. The quantizer_type argument for the decorator should be of type of the `TrainingMethod  enum. See explaination about `@mark_quantizer` and how to use it under the [Kears Quantization Infrastructure](../../quantizers_infrastructure/keras/README.md).
   
## Example: Symmetric Weights Quantizer
To create custom `MyWeightsTrainingQuantizer` which is a symmetric weights training quantizer you need to set
`qi.QuantizationTarget.Weights` as target and `qi.QuantizationMethod.SYMMETRIC` as method.
Assume that the quantizer has a new training method called `MyTrainig` which is defined in the `TrainingMethod` Enum.
```python
import tensorflow as tf
from model_compression_toolkit import quantizers_infrastructure as qi, TrainingMethod
from model_compression_toolkit.quantizers_infrastructure.common.base_inferable_quantizer import mark_quantizer
from model_compression_toolkit.core.common.target_platform import QuantizationMethod
from model_compression_toolkit.qat.keras.quantizer.base_keras_qat_quantizer import BaseKerasQATTrainableQuantizer
NEW_PARAM = "new_param_name"

@mark_quantizer(quantization_target=qi.QuantizationTarget.Weights,
                quantization_method=[QuantizationMethod.SYMMETRIC],
                quantizer_type=TrainingMethod.MyTraining)
class MyWeightsTrainingQuantizer(BaseKerasQATTrainableQuantizer):
    def __init__(self, quantization_config: NodeWeightsQuantizationConfig):
        super(MyWeightsTrainingQuantizer, self).__init__(quantization_config)
        # Define your new params here:
        self.new_param = ...
    
    def initialize_quantization(self, tensor_shape, name, layer):
        # Creating new params for quantizer
        new_param = layer.add_weight(NEW_PARAM,
                                     shape=len(tensor_shape),
                                     initializer=tf.keras.initializers.Constant(1.0),
                                     trainable=True)
        new_param.assign(self.new_param)
        # Save the quantizer parameters for later calculations
        self.quantizer_parameters = {NEW_PARAM: new_param}
        return self.quantizer_parameters
    
    def __call__(self, inputs):
        # Your quantization logic here
        new_param = self.quantizer_parameters[NEW_PARAM]
        # Custom quantization function you need to implement
        quantized_inputs = custom_quantize(inputs, new_param)
        return quantized_inputs

    def convert2inferable(self):
        return qi.WeightsUniformInferableQuantizer(...)
```

## Example: Symmetric Activations Quantizer
To create custom `MyActivationsTrainingQuantizer` which is a symmetric activations training quantizer you need to set `qi.QuantizationTarget.Activation` as target and `qi.QuantizationMethod.SYMMETRIC` as method.
Assume that the quantizer has a new training method called `MyTrainig` which is defined in the `TrainingMethod` Enum.
```python
import tensorflow as tf
NEW_PARAM = "new_param_name"
from model_compression_toolkit import quantizers_infrastructure as qi, TrainingMethod
from model_compression_toolkit.quantizers_infrastructure.common.base_inferable_quantizer import mark_quantizer
from model_compression_toolkit.core.common.target_platform import QuantizationMethod
from model_compression_toolkit.qat.keras.quantizer.base_keras_qat_quantizer import BaseKerasQATTrainableQuantizer

@mark_quantizer(quantization_target=qi.QuantizationTarget.Activation,
                quantization_method=[QuantizationMethod.SYMMETRIC],
                quantizer_type=TrainingMethod.TrainingMethod)
class MyActivationsTrainingQuantizer(BaseKerasQATTrainableQuantizer):
    def __init__(self, quantization_config: NodeActivationQuantizationConfig):
        super(MyActivationsTrainingQuantizer, self).__init__(quantization_config)
        # Define your new params here:
        self.new_param = ...
    
    def initialize_quantization(self, tensor_shape, name, layer):
        # Creating new params for quantizer
        new_param = layer.add_weight(NEW_PARAM,
                                     shape=len(tensor_shape),
                                     initializer=tf.keras.initializers.Constant(1.0),
                                     trainable=True)
        new_param.assign(self.new_param)
        # Save the quantizer parameters for later calculations
        self.quantizer_parameters = {NEW_PARAM: new_param}
        return self.quantizer_parameters
    
    def __call__(self, inputs):
        # Your quantization logic here
        new_param = self.quantizer_parameters[NEW_PARAM]
        # Custom quantization function you need to implement
        quantized_inputs = custom_quantize(inputs, new_param)
        return quantized_inputs

    def convert2inferable(self):
        return qi.ActivationUniformInferableQuantizer(...)
```