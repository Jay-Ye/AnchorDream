"""
Utility functions for modifying state dictionaries to handle model architecture changes.
"""

import torch
import torch.nn as nn
import math
from typing import Dict, Any, Tuple
from imaginaire.utils import log


def expand_patch_embed_weights_in_state_dict(
    state_dict: Dict[str, torch.Tensor],
    old_in_channels: int,
    new_in_channels: int,
    spatial_patch_size: int,
    temporal_patch_size: int,
    weight_key: str = "x_embedder.proj.1.weight",
) -> Dict[str, torch.Tensor]:
    """
    Expand the patch embedder weights in a state dictionary by duplicating them along the input channel dimension.
    
    Args:
        state_dict (Dict[str, torch.Tensor]): The state dictionary to modify
        old_in_channels (int): Original number of input channels
        new_in_channels (int): New number of input channels
        spatial_patch_size (int): Spatial patch size
        temporal_patch_size (int): Temporal patch size
        weight_key (str): Key for the patch embedder weight in the state dict
        
    Returns:
        Dict[str, torch.Tensor]: Modified state dictionary with expanded weights
    """
    if new_in_channels < old_in_channels:
        raise ValueError(f"new_in_channels ({new_in_channels}) must be >= old_in_channels ({old_in_channels})")
    
    # Check if the weight key exists in the state dict
    if weight_key not in state_dict:
        log.warning(f"Weight key '{weight_key}' not found in state dict. Available keys: {list(state_dict.keys())[:10]}...")
        return state_dict
    
    # Get the original weight
    old_weight = state_dict[weight_key]  # shape: (out_channels, old_in_channels * patch_size^3)
    
    # Calculate patch size
    patch_size = spatial_patch_size * spatial_patch_size * temporal_patch_size
    
    # Verify the weight shape matches expectations
    expected_old_dim = old_in_channels * patch_size
    if old_weight.shape[1] != expected_old_dim:
        log.warning(f"Expected weight shape[1] to be {expected_old_dim}, but got {old_weight.shape[1]}. "
                   f"This might indicate different patch sizes or channel configuration.")
        # Try to infer the actual old_in_channels from the weight shape
        inferred_old_in_channels = old_weight.shape[1] // patch_size
        if old_weight.shape[1] % patch_size != 0:
            raise ValueError(f"Weight shape[1] ({old_weight.shape[1]}) is not divisible by patch_size ({patch_size})")
        log.info(f"Inferred old_in_channels: {inferred_old_in_channels} (from weight shape)")
        old_in_channels = inferred_old_in_channels
    
    out_channels = old_weight.shape[0]
    dim = new_in_channels * patch_size

    # Reshape to separate channel dimension from patch dimensions
    old_weight_reshaped = old_weight.view(out_channels, old_in_channels, patch_size)  # (out_channels, old_in_channels, patch_size)
    new_weight = torch.zeros(out_channels, new_in_channels, patch_size, dtype=old_weight.dtype, device=old_weight.device)

    # last 2 channels are for condition mask and padding mask
    new_weight[:, :old_in_channels-2, :] = old_weight_reshaped[:, :old_in_channels-2, :]
    new_weight[:, old_in_channels-2:-2, :] = old_weight_reshaped[:, :old_in_channels-2, :]
    new_weight[:, -2:, :] = old_weight_reshaped[:, -2:, :]

    # Reshape back to the linear layer format
    new_weight = new_weight.view(out_channels, dim)
    state_dict[weight_key] = new_weight
    
    log.info(f"Expanded patch embedder weights from {old_in_channels} to {new_in_channels} channels "
             f"(extend_mode: duplicate)")
    
    return state_dict


def detect_and_expand_patch_embed_weights(
    state_dict: Dict[str, torch.Tensor],
    model_in_channels: int,
    spatial_patch_size: int,
    temporal_patch_size: int,
    weight_key_patterns: list[str] = None
) -> Tuple[Dict[str, torch.Tensor], bool]:
    """
    Automatically detect and expand patch embedder weights if needed.
    
    Args:
        state_dict (Dict[str, torch.Tensor]): The state dictionary to check and modify
        model_in_channels (int): Expected number of input channels in the model
        spatial_patch_size (int): Spatial patch size
        temporal_patch_size (int): Temporal patch size
        weight_key_patterns (list[str], optional): Patterns to search for patch embedder weights.
            Defaults to common patterns.
            
    Returns:
        Tuple[Dict[str, torch.Tensor], bool]: (modified_state_dict, was_expanded)
    """
    if weight_key_patterns is None:
        weight_key_patterns = [
            "x_embedder.proj.1.weight",
            "net.x_embedder.proj.1.weight",
            "model.x_embedder.proj.1.weight",
            "dit.x_embedder.proj.1.weight"
        ]
    
    # Find the actual weight key in the state dict
    weight_key = None
    for pattern in weight_key_patterns:
        if pattern in state_dict:
            weight_key = pattern
            break
    
    if weight_key is None:
        log.warning("No patch embedder weight found in state dict. Available keys: "
                   f"{list(state_dict.keys())[:10]}...")
        return state_dict, False
    
    # Get the weight and infer the original number of channels
    weight = state_dict[weight_key]
    patch_size = spatial_patch_size * spatial_patch_size * temporal_patch_size
    
    if weight.shape[1] % patch_size != 0:
        log.warning(f"Weight shape[1] ({weight.shape[1]}) is not divisible by patch_size ({patch_size})")
        return state_dict, False
    
    old_in_channels = weight.shape[1] // patch_size
    
    # Check if expansion is needed
    if old_in_channels == model_in_channels:
        log.info(f"Patch embedder weights already match model in_channels ({model_in_channels})")
        return state_dict, False
    
    if old_in_channels > model_in_channels:
        log.warning(f"State dict has more channels ({old_in_channels}) than model expects ({model_in_channels}). "
                   f"This might cause issues.")
        return state_dict, False
    
    # Expand the weights
    log.info(f"Detected mismatch: state dict has {old_in_channels} channels, model expects {model_in_channels} channels")
    expanded_state_dict = expand_patch_embed_weights_in_state_dict(
        state_dict, old_in_channels, model_in_channels, spatial_patch_size, temporal_patch_size, weight_key
    )
    
    return expanded_state_dict, True


def get_model_patch_embed_info(model: nn.Module) -> Tuple[int, int, int]:
    """
    Extract patch embedder information from a model.
    
    Args:
        model (nn.Module): The model to analyze
        
    Returns:
        Tuple[int, int, int]: (in_channels, spatial_patch_size, temporal_patch_size)
    """
    if hasattr(model, 'x_embedder'):
        patch_embed = model.x_embedder
        if hasattr(patch_embed, 'spatial_patch_size') and hasattr(patch_embed, 'temporal_patch_size'):
            # Calculate in_channels from the weight shape
            weight = patch_embed.proj[1].weight
            patch_size = patch_embed.spatial_patch_size * patch_embed.spatial_patch_size * patch_embed.temporal_patch_size
            in_channels = weight.shape[1] // patch_size
            return in_channels, patch_embed.spatial_patch_size, patch_embed.temporal_patch_size
    
    # Fallback: try to get from model attributes
    if hasattr(model, 'in_channels') and hasattr(model, 'patch_spatial') and hasattr(model, 'patch_temporal'):
        return model.in_channels, model.patch_spatial, model.patch_temporal
    
    raise ValueError("Could not extract patch embedder information from model") 