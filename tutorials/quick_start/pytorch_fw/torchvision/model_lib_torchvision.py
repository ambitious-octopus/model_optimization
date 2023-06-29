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
import torch
import torchvision
from torchvision.models import list_models, get_model, get_model_weights, get_weight
from torch.utils.data import Subset

from common.model_lib import BaseModelLib
from pytorch_fw.utils import classification_eval, get_representative_dataset
from common.constants import MODEL_NAME, BATCH_SIZE, VALIDATION_SET_LIMIT, VALIDATION_DATASET_FOLDER, IMAGENET_DATASET

from common.results import DatasetInfo


class ModelLib(BaseModelLib):

    @staticmethod
    def get_torchvision_model(model_name):
        all_models = list_models() # List all torchvision models
        if model_name in all_models:
            # Initialize model with the best available weights
            return get_model(model_name, weights="DEFAULT")
        else:
            raise Exception(f'Unknown torchvision model name {model_name}, Please check available models in https://pytorch.org/vision/stable/models.html')

    @staticmethod
    def get_torchvision_weights(model_name):

        # Return the best available weights of the model
        return get_model_weights(model_name).DEFAULT

    def __init__(self, args):
        self.model = self.get_torchvision_model(args[MODEL_NAME])
        self.preprocess = self.get_torchvision_weights(args[MODEL_NAME]).transforms()
        self.dataset_name = IMAGENET_DATASET
        super().__init__(args)

    def get_model(self):
        return self.model

    def get_representative_dataset(self, representative_dataset_folder, n_iter, batch_size):
        ds = torchvision.datasets.ImageFolder(representative_dataset_folder, transform=self.preprocess)
        dl = torch.utils.data.DataLoader(ds, batch_size, shuffle=True)
        return get_representative_dataset(dl, n_iter)

    def evaluate(self, model):
        batch_size = int(self.args[BATCH_SIZE])
        validation_dataset_folder = self.args[VALIDATION_DATASET_FOLDER]
        testset = torchvision.datasets.ImageFolder(validation_dataset_folder, transform=self.preprocess)
        testloader = torch.utils.data.DataLoader(testset,
                                                  batch_size=batch_size,
                                                  shuffle=False)
        acc, total = classification_eval(model, testloader, self.args[VALIDATION_SET_LIMIT])
        dataset_info = DatasetInfo(self.dataset_name, total)
        return acc, dataset_info


