# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Registry for unsloth-custom MLX model loaders.

Extension point for the loading priority chain in mlx_loader.py:
  1. Check unsloth custom loader registry (this module)
  2. Check if VLM → mlx_vlm.load()
  3. Fallback → mlx_lm.load()
"""

_UNSLOTH_LOADERS: dict = {}


def get_unsloth_loader(model_type):
    """Return the custom loader for a model_type, or None if not registered."""
    return _UNSLOTH_LOADERS.get(model_type)


def register_loader(model_type, loader_fn):
    """Register a custom loader callable for a model_type."""
    _UNSLOTH_LOADERS[model_type] = loader_fn
