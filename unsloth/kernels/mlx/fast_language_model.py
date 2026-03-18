# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
FastLanguageModel and FastModel for the pure MLX path.

These provide API compatibility with unsloth's GPU FastLanguageModel but
route entirely through MLX for Apple Silicon training.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple, Any

import mlx.core as mx


def _resolve_dtype(dtype_str: Optional[str]) -> type:
    """Map dtype string to MLX dtype."""
    if dtype_str is None:
        return mx.float16
    mapping = {
        "float32": mx.float32,
        "float16": mx.float16,
        "bfloat16": mx.bfloat16,
    }
    return mapping.get(str(dtype_str).lower(), mx.float16)


class FastLanguageModel:
    """
    Pure MLX model loader, compatible with the unsloth FastLanguageModel API.

    Usage:
        model, tokenizer = FastLanguageModel.from_pretrained(
            "unsloth/Llama-3.2-1B-Instruct",
            max_seq_length=2048,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(model, r=16, ...)
    """

    @staticmethod
    def from_pretrained(
        model_name: str = "unsloth/Llama-3.2-1B-Instruct",
        max_seq_length: int = 2048,
        dtype=None,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        load_in_16bit: bool = False,
        full_finetuning: bool = False,
        token: Optional[str] = None,
        trust_remote_code: bool = False,
        **kwargs,
    ) -> Tuple[Any, Any]:
        """
        Load a pretrained model using MLX backend.

        Returns (model, tokenizer) tuple.
        """
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

        # Login if token provided
        if token:
            try:
                from huggingface_hub import login
                login(token=token)
            except Exception:
                pass

        print(f"Unsloth: Loading {model_name} with pure MLX backend...")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            token=token,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Determine torch dtype for initial loading
        if dtype is None:
            torch_dtype = torch.float16
        elif isinstance(dtype, str):
            torch_dtype = getattr(torch, dtype, torch.float16)
        else:
            torch_dtype = dtype

        # Load model config
        config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            token=token,
        )

        # Load the PyTorch model (will be bridged to MLX for compute)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="cpu",  # Load to CPU, MLX bridge handles compute
            trust_remote_code=trust_remote_code,
            token=token,
        )

        # Store metadata for LoRA and training
        model.max_seq_length = max_seq_length
        model._mlx_dtype = _resolve_dtype(
            str(torch_dtype).replace("torch.", "") if torch_dtype else None
        )
        model._load_in_4bit = load_in_4bit
        model._full_finetuning = full_finetuning

        # Try to load MLX quantized weights if available
        if load_in_4bit:
            from .loader import load_mlx_weights
            try:
                from huggingface_hub import hf_hub_download, list_repo_files
                repo_files = list_repo_files(model_name, token=token)
                mlx_files = [f for f in repo_files if f.endswith(".safetensors")]
                for f in mlx_files:
                    path = hf_hub_download(model_name, f, token=token)
                    if load_mlx_weights(model, path, config=config):
                        print(f"Unsloth: Loaded MLX weights from {f}")
                        break
            except Exception:
                pass

        print(
            f"Unsloth: {model_name} loaded.\n"
            f"   MLX dtype: {model._mlx_dtype}\n"
            f"   Max seq length: {max_seq_length}\n"
            f"   4-bit quantization: {load_in_4bit}"
        )

        return model, tokenizer

    @staticmethod
    def get_peft_model(
        model,
        r: int = 16,
        target_modules=None,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        bias: str = "none",
        use_gradient_checkpointing: Any = "unsloth",
        random_state: int = 3407,
        use_rslora: bool = False,
        max_seq_length: int = 2048,
        **kwargs,
    ):
        """
        Apply LoRA adapters using MLX.

        Returns the model with LoRA layers applied.
        """
        from .lora import LoRAConfig, get_peft_model

        if target_modules is None:
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]

        lora_config = LoRAConfig(
            r=r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            target_modules=target_modules,
        )

        model = get_peft_model(model, lora_config)

        print(
            f"Unsloth: Applied LoRA with r={r}, alpha={lora_alpha}\n"
            f"   Target modules: {target_modules}"
        )

        return model

    @staticmethod
    def for_inference(model):
        """Prepare model for inference (merge LoRA weights)."""
        from .merge_lora import mlx_merge_lora
        try:
            mlx_merge_lora(model)
        except Exception:
            pass
        return model


# FastModel is an alias — on MLX, there's no distinction between
# the "base" and "language model" loaders.
FastModel = FastLanguageModel
