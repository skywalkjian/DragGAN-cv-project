import torch
import torch.nn.functional as F
import numpy as np

class MaskHandler:
    def __init__(self):
        self.name = "Base"

    def handle_mask(self, feat, feat0, mask, **kwargs):
        """
        Returns:
            loss: scalar tensor for background preservation
            modified_feat: (optional) modified feature map for tracking/supervision
        """
        return 0, feat

class BaselineMaskHandler(MaskHandler):
    def __init__(self):
        super().__init__()
        self.name = "Baseline (L1 Loss)"

    def handle_mask(self, feat, feat0, mask, lambda_mask=10, **kwargs):
        if mask is None:
            return 0, feat
        
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).to(feat.device).float()
        
        # In DragGAN, mask=1 usually means FIXED region (background to preserve)
        # and mask=0 means MOVABLE region. 
        # Let's check the min/max to be safe.
        if mask.min() == 0 and mask.max() == 1:
            mask_usq = mask.unsqueeze(0).unsqueeze(0)
            # Resize mask to feature resolution if needed
            if mask_usq.shape[-2:] != feat.shape[-2:]:
                mask_usq = F.interpolate(mask_usq, feat.shape[-2:], mode='bilinear')
            
            loss_fix = F.l1_loss(feat * mask_usq, feat0 * mask_usq)
            return lambda_mask * loss_fix, feat
        
        return 0, feat

class FeatureBlendingMaskHandler(MaskHandler):
    def __init__(self, blend_alpha=0.8, feather_sigma=3):
        super().__init__()
        self.name = "Feature Blending (Robust)"
        self.blend_alpha = blend_alpha
        self.feather_sigma = feather_sigma

    def _feather_mask(self, mask, sigma):
        if sigma <= 0:
            return mask
        # Simple Gaussian blur for feathering
        kernel_size = int(2 * round(3 * sigma) + 1)
        # Create 1D Gaussian kernel
        x = torch.arange(kernel_size).float() - (kernel_size - 1) / 2
        kernel_1d = torch.exp(-x.pow(2) / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        # Expand to 2D
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        kernel_2d = kernel_2d.expand(1, 1, kernel_size, kernel_size).to(mask.device)
        
        # Apply padding and conv
        pad = kernel_size // 2
        mask_padded = F.pad(mask, (pad, pad, pad, pad), mode='replicate')
        feathered = F.conv2d(mask_padded, kernel_2d)
        return feathered

    def handle_mask(self, feat, feat0, mask, lambda_mask=10, **kwargs):
        if mask is None:
            return 0, feat
        
        device = feat.device
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).to(device).float()
        
        mask_usq = mask.unsqueeze(0).unsqueeze(0)
        
        # 1. Resize and Feather Mask
        if mask_usq.shape[-2:] != feat.shape[-2:]:
            mask_usq = F.interpolate(mask_usq, feat.shape[-2:], mode='bilinear')
        
        soft_mask = self._feather_mask(mask_usq, self.feather_sigma)
        
        # 2. Feature Blending (Hard Injection for background)
        # F_fused = M * F_0 + (1-M) * F_t  (Assuming M=1 for background)
        # We use a blend_alpha to control the strength of injection vs optimization
        modified_feat = soft_mask * feat0 + (1 - soft_mask) * feat
        
        # 3. Blending Loss (Soft constraint on the transition area)
        # We still want the latent space to try to match the background
        loss_fix = F.l1_loss(feat * soft_mask, feat0 * soft_mask)
        
        return lambda_mask * loss_fix, modified_feat

class SchedulingMaskHandler(MaskHandler):
    def __init__(self, start_lambda=1.0, end_lambda=20.0, warm_up_steps=20):
        super().__init__()
        self.name = "Loss Scheduling (Adaptive)"
        self.start_lambda = start_lambda
        self.end_lambda = end_lambda
        self.warm_up_steps = warm_up_steps
        self.current_step = 0

    def handle_mask(self, feat, feat0, mask, lambda_mask=10, **kwargs):
        if mask is None:
            return 0, feat
        
        # Calculate scheduled lambda
        if self.current_step < self.warm_up_steps:
            alpha = self.current_step / self.warm_up_steps
            curr_lambda = self.start_lambda + alpha * (self.end_lambda - self.start_lambda)
        else:
            curr_lambda = self.end_lambda
        
        self.current_step += 1
        
        device = feat.device
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).to(device).float()
            
        mask_usq = mask.unsqueeze(0).unsqueeze(0)
        if mask_usq.shape[-2:] != feat.shape[-2:]:
            mask_usq = F.interpolate(mask_usq, feat.shape[-2:], mode='bilinear')
        
        loss_fix = F.l1_loss(feat * mask_usq, feat0 * mask_usq)
        return curr_lambda * loss_fix, feat

    def reset(self):
        self.current_step = 0

def get_mask_handler(name):
    if name == 'DragGAN (Baseline)':
        return BaselineMaskHandler()
    elif name == 'Feature Blending (Robust)':
        return FeatureBlendingMaskHandler(blend_alpha=1.0, feather_sigma=2.0)
    elif name == 'Loss Scheduling (Adaptive)':
        return SchedulingMaskHandler(start_lambda=0.0, end_lambda=15.0, warm_up_steps=10)
    else:
        return BaselineMaskHandler()
