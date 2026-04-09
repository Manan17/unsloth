# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
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

import os, platform, importlib.util

os.environ["UNSLOTH_IS_PRESENT"] = "1"

# Detect Apple Silicon + MLX before any torch/numpy imports
_IS_MLX = (
    platform.system() == "Darwin"
    and platform.machine() == "arm64"
    and importlib.util.find_spec("mlx") is not None
)

if _IS_MLX:
    import unsloth_zoo
    from unsloth_zoo.mlx_trainer import MLXTrainer, MLXTrainingConfig
    from unsloth_zoo.mlx_trainer import train_on_responses_only
    from unsloth_zoo.mlx_loader import FastMLXModel
    from unsloth_zoo.dataset_utils import standardize_data_formats

    standardize_sharegpt = standardize_data_formats
    __version__ = unsloth_zoo.__version__
else:
    # Full GPU init: torch, triton, bitsandbytes, model patches, etc.
    from ._gpu_init import *
    from ._gpu_init import __version__


class FastLanguageModel:
    """Unified entry point for loading and fine-tuning LLMs.

    Automatically routes to the MLX backend on Apple Silicon
    or the CUDA/GPU backend elsewhere.
    """

    @staticmethod
    def from_pretrained(*args, **kwargs):
        if _IS_MLX:
            return FastMLXModel.from_pretrained(*args, **kwargs)
        else:
            from .models.loader import FastLanguageModel as _GPU

            return _GPU.from_pretrained(*args, **kwargs)

    @staticmethod
    def get_peft_model(*args, **kwargs):
        if _IS_MLX:
            return FastMLXModel.get_peft_model(*args, **kwargs)
        else:
            from .models.loader import FastLanguageModel as _GPU

            return _GPU.get_peft_model(*args, **kwargs)

    @staticmethod
    def for_inference(*args, **kwargs):
        if _IS_MLX:
            model = args[0] if args else kwargs.get("model")
            if model is not None:
                model.eval()
            return model
        else:
            from .models.loader import FastLanguageModel as _GPU

            return _GPU.for_inference(*args, **kwargs)

    @staticmethod
    def for_training(*args, **kwargs):
        if _IS_MLX:
            model = args[0] if args else kwargs.get("model")
            if model is not None:
                model.train()
            return model
        else:
            from .models.loader import FastLanguageModel as _GPU

            return _GPU.for_training(*args, **kwargs)


FastModel = FastLanguageModel
FastVisionModel = FastModel
