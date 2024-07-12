import sys
sys.path.append('../')
import os
import model_compression_toolkit as mct
from tutorials.mct_model_garden.evaluation_metrics.coco_evaluation import coco_dataset_generator
from tutorials.mct_model_garden.models_pytorch.yolov8.yolov8_preprocess import yolov8_preprocess_chw_transpose
from typing import Iterator, Tuple, List
import wget 
import zipfile
import logging
import coloredlogs
coloredlogs.install()
logging.basicConfig(level=logging.INFO)


if not os.path.isdir('coco'):
    logging.info('Downloading COCO dataset')
    os.mkdir('coco')
    wget.download('http://images.cocodataset.org/annotations/annotations_trainval2017.zip')
    with zipfile.ZipFile("annotations_trainval2017.zip", 'r') as zip_ref:
        zip_ref.extractall('./coco')
    os.remove('annotations_trainval2017.zip')
    
    wget.download('http://images.cocodataset.org/zips/val2017.zip')
    with zipfile.ZipFile("val2017.zip", 'r') as zip_ref:
        zip_ref.extractall('./coco')
    os.remove('val2017.zip')
    

from ultralytics import YOLO
model = YOLO('yolov8n.yaml').load("yolov8n.pt").model

REPRESENTATIVE_DATASET_FOLDER = './coco/val2017/'
REPRESENTATIVE_DATASET_ANNOTATION_FILE = './coco/annotations/instances_val2017.json'
BATCH_SIZE = 4
n_iters = 20

# Load representative dataset
logging.info('Loading representative dataset')
representative_dataset = coco_dataset_generator(dataset_folder=REPRESENTATIVE_DATASET_FOLDER,
                                                annotation_file=REPRESENTATIVE_DATASET_ANNOTATION_FILE,
                                                preprocess=yolov8_preprocess_chw_transpose,
                                                batch_size=BATCH_SIZE)

# Define representative dataset generator
def get_representative_dataset(n_iter: int, dataset_loader: Iterator[Tuple]):
    """
    This function creates a representative dataset generator. The generator yields numpy
        arrays of batches of shape: [Batch, H, W ,C].
    Args:
        n_iter: number of iterations for MCT to calibrate on
    Returns:
        A representative dataset generator
    """       
    def representative_dataset() -> Iterator[List]:
        ds_iter = iter(dataset_loader)
        for _ in range(n_iter):
            yield [next(ds_iter)[0]]

    return representative_dataset

logging.info('Creating representative dataset generator')
# Get representative dataset generator
representative_dataset_gen = get_representative_dataset(n_iter=n_iters,
                                                        dataset_loader=representative_dataset)

# Set IMX500-v1 TPC
logging.info('Setting target platform capabilities')
tpc = mct.get_target_platform_capabilities(fw_name="pytorch",
                                           target_platform_name='imx500',
                                           target_platform_version='v1')


# # Specify the necessary configuration for mixed precision quantization. To keep the tutorial brief, we'll use a small set of images and omit the hessian metric for mixed precision calculations. It's important to be aware that this choice may impact the resulting accuracy. 
mp_config = mct.core.MixedPrecisionQuantizationConfig(num_of_images=5,
                                                      use_hessian_based_scores=False)
config = mct.core.CoreConfig(mixed_precision_config=mp_config,
                             quantization_config=mct.core.QuantizationConfig(shift_negative_activation_correction=True))

# # Define target Resource Utilization for mixed precision weights quantization (75% of 'standard' 8bits quantization)
resource_utilization_data = mct.core.pytorch_resource_utilization_data(in_model=model,
                                                                       representative_data_gen=
                                                                       representative_dataset_gen,
                                                                       core_config=config,
                                                                       target_platform_capabilities=tpc)


resource_utilization = mct.core.ResourceUtilization(weights_memory=resource_utilization_data.weights_memory * 0.75)

# Perform post training quantization
quant_model, _ = mct.ptq.pytorch_post_training_quantization(in_module=model,
                                                            representative_data_gen=
                                                            representative_dataset_gen,
                                                            # target_resource_utilization=resource_utilization,
                                                            core_config=config,
                                                            target_platform_capabilities=tpc)
print('Quantized model is ready')