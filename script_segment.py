import os
import cv2  # type: ignore
import torch
from typing import Any, Optional
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from rotary_embedding_torch import RotaryEmbedding
import torch.nn.functional as F
from PIL import Image
import numpy as np

def load_sam_model(model_type: str = 'vit_b',
                   checkpoint: str = 'weights/sam_vit_b_01ec64.pth',
                   device: Optional[str] = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    return sam.to(device=device)


def generate_masks_from_batch(
    model,
    input_tensor: torch.Tensor,
    output_mode: str = "binary_mask",
    save_masks_dir: Optional[str] = None,
    upscale: bool = False,
    scale_factor: int = 2,
    interp_mode: str = 'bilinear',   # 'nearest' | 'bilinear' | 'bicubic'
    apply_activation: bool = False,
    **amg_kwargs: Any
) -> torch.Tensor:
    """
    - upscale: whether to first scale up the input
    - scale_factor: scaling factor (e.g., 2)
    - interp_mode: interpolation method
    - apply_activation: whether to apply thresholding after pooling
    """
    with torch.no_grad():
        device = input_tensor.device
        B, C, H, W = input_tensor.shape

        # 1) Optional interpolation upscaling
        if upscale:
            tensor_up = F.interpolate(
                input_tensor,
                scale_factor=scale_factor,
                mode=interp_mode,
                align_corners=(interp_mode!='nearest')
            )
        else:
            tensor_up = input_tensor
        _, _, H2, W2 = tensor_up.shape

        # 2) Convert to numpy BGR for SAM
        min_val, max_val = tensor_up.min().item(), tensor_up.max().item()
        imgs = ((tensor_up - min_val) / (max_val - min_val + 1e-6) * 255
                ).clamp(0, 255).byte()
        imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()  # (B, H2, W2, C)

        # 3) Instantiate SAM mask generators
        mask_gen = SamAutomaticMaskGenerator(
            model=model,
            output_mode=output_mode,
            **amg_kwargs
        )

        # A second, sparser mask generator for coarser segmentation
        mask_gen_v2 = SamAutomaticMaskGenerator(
            model=model,
            points_per_side=32,            # sparser grid → coarse masks
            pred_iou_thresh=0.96,          # drop low-confidence masks
            stability_score_thresh=0.97,   # keep only the most stable masks          
            output_mode=output_mode,
            **amg_kwargs
        )

        all_masks = []
        max_masks = 0
        batch_masks = []

        # 3.1) Generate raw masks per image
        for b, img_rgb in enumerate(imgs):
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            masks = mask_gen.generate(img_bgr)  # list of dicts, each contains "segmentation"
            batch_masks.append(masks)
            max_masks = max(max_masks, len(masks))

            # Optionally save individual masks
            if save_masks_dir:
                d = os.path.join(save_masks_dir, f"image_{b}")
                os.makedirs(d, exist_ok=True)
                for i, m in enumerate(masks):
                    seg8 = (m["segmentation"].astype(np.uint8) * 255)
                    slice_rgb = cv2.bitwise_and(img_rgb, img_rgb, mask=seg8)
                    Image.fromarray(slice_rgb).save(os.path.join(d, f"mask_{i}.png"))

        # 4) Stack masks into tensors and pad to uniform count
        for masks in batch_masks:
            if masks:
                m_tensors = [torch.tensor(m["segmentation"], dtype=torch.float32, device=device)
                             for m in masks]
                m_stack = torch.stack(m_tensors, dim=0)  # (M, H2, W2)
                if m_stack.size(0) < max_masks:
                    pad_cnt = max_masks - m_stack.size(0)
                    pad = torch.zeros((pad_cnt, H2, W2), device=device)
                    m_stack = torch.cat([m_stack, pad], dim=0)
                # Background mask as inverse of union of all masks
                no_mask = torch.ones((H2, W2), device=device)
                for i in range(len(masks)):
                    no_mask *= (1 - m_stack[i])
            else:
                # No masks: all zeros + full background mask
                m_stack = torch.zeros((max_masks, H2, W2), device=device)
                no_mask = torch.ones((H2, W2), device=device)

            # Combine foreground and background → (M+1, H2, W2)
            stacked = torch.cat([m_stack, no_mask.unsqueeze(0)], dim=0)
            all_masks.append(stacked.unsqueeze(0))  # (1, M+1, H2, W2)

        masks_bchw = torch.cat(all_masks, dim=0)  # (B, M+1, H2, W2)

        # 5) Downscale back to original size via average pooling
        if upscale:
            masks_bchw = F.avg_pool2d(
                masks_bchw,
                kernel_size=scale_factor,
                stride=scale_factor
            )  # → (B, M+1, H, W)

        # 6) Optional binary activation
        if apply_activation:
            masks_bchw = (masks_bchw >= 0.5).float()

        return masks_bchw


def generate_weighted_noise(masks_tensor: torch.Tensor) -> torch.Tensor:
    """
    Generate weighted RGB noise across GPUs based on masks.

    Args:
        masks_tensor: Mask tensor of shape (B, M, H, W).

    Returns:
        Weighted noise tensor of shape (B, 3, H, W).
    """
    with torch.no_grad():
        device = masks_tensor.device
        B, M, H, W = masks_tensor.shape

        weighted_noise = torch.zeros((B, 3, H, W), dtype=torch.float32, device=device)
        coverage_count = torch.zeros((B, 1, H, W), dtype=torch.float32, device=device)

        for b in range(B):
            for m in range(M):
                mask = masks_tensor[b, m].unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

                # Generate independent noise per channel
                noise_rgb = torch.randn((1, 3, H, W), dtype=torch.float32, device=device)
                masked_noise = noise_rgb * mask

                # Accumulate weighted noise
                weighted_noise[b] += masked_noise[0]
                coverage_count[b] += mask[0]

        # Avoid division by zero and average
        coverage_count = torch.maximum(coverage_count, torch.ones_like(coverage_count))
        weighted_noise /= coverage_count

        return weighted_noise


def count_masks_per_pixel(masks_tensor: torch.Tensor) -> torch.Tensor:
    """
    Count how many masks cover each pixel.

    Args:
        masks_tensor: Mask tensor of shape (B, M, H, W).

    Returns:
        Tensor of shape (B, H, W) with per-pixel mask counts.
    """
    with torch.no_grad():
        mask_count = masks_tensor.sum(dim=1)
        return mask_count


def merge_all_mask_to_one_RoPE_batch(all_masks: torch.Tensor) -> torch.Tensor:
    """
    Merge a batch of masks into a single mask using Rotary Position Embedding (RoPE).

    Args:
        all_masks: Tensor of shape (B, M, W, H).

    Returns:
        Merged mask tensor of shape (B, 1, W, H).
    """
    with torch.no_grad():
        B, M, W, H = all_masks.shape
        device = all_masks.device

        # Initialize RotaryEmbedding
        rotary_emb = RotaryEmbedding(dim=H).to(device)

        # Original mask embedding
        mask_embed_ori = torch.ones((B, 1, W, H), dtype=torch.float32, device=device)
        mask_embed_ori = rotary_emb.rotate_queries_or_keys(mask_embed_ori)

        merge_masks = torch.zeros((B, 1, W, H), dtype=torch.float32, device=device)

        for m in range(M):
            mask = all_masks[:, m, :, :].unsqueeze(1)  # (B, 1, W, H)
            # Compute average embedding for this mask
            mask_embed_num = (mask_embed_ori * mask).mean(dim=[2, 3], keepdim=True)
            mask_embed = torch.ones((B, 1, W, H), dtype=torch.float32, device=device) * mask_embed_num
            # Merge: later masks overwrite earlier ones
            merge_masks = torch.where(mask.bool(), mask_embed, merge_masks)

        return merge_masks


def normalize_noise_to_unit_variance(noise_tensor: torch.Tensor) -> torch.Tensor:
    """
    Normalize noise so its variance is 1.

    Args:
        noise_tensor: Noise tensor of shape (B, C, H, W).

    Returns:
        Noise tensor normalized to unit variance.
    """
    if not noise_tensor.dtype.is_floating_point:
        noise_tensor = noise_tensor.float()

    # Compute per-sample, per-channel mean and std
    mean = noise_tensor.mean(dim=[2, 3], keepdim=True)
    std = noise_tensor.std(dim=[2, 3], keepdim=True)

    # Prevent division by zero
    std = torch.clamp(std, min=1e-6)

    normalized_noise = (noise_tensor - mean) / std
    return normalized_noise
