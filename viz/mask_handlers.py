import torch
import torch.nn.functional as F
import numpy as np

class MaskHandler:
    """Base class for handling background preservation masks."""
    def __init__(self):
        self.name = "Base"

    def handle_mask(self, feat, feat0, mask, **kwargs):
        return 0.0, feat

    def reset(self):
        """Reset internal state."""
        pass

class BaselineMaskHandler(MaskHandler):
    """L1 loss based background preservation."""
    def __init__(self):
        super().__init__()
        self.name = "Baseline"

    def handle_mask(self, feat, feat0, mask, lambda_mask=10, **kwargs):
        if mask is None:
            return 0.0, feat
        
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).to(feat.device).float()
        
        mask_usq = mask.unsqueeze(0).unsqueeze(0)
        if mask_usq.shape[-2:] != feat.shape[-2:]:
            mask_usq = F.interpolate(mask_usq, feat.shape[-2:], mode='bilinear')
        
        # User paints flexible area (mask=1), we fix background (mask=0)
        fixed_mask = 1.0 - mask_usq
        loss_fix = F.l1_loss(feat * fixed_mask, feat0 * fixed_mask)
        return lambda_mask * loss_fix, feat

class LossSchedulingMaskHandler(MaskHandler):
    """Adaptive background loss with temporal scheduling."""
    def __init__(self, start_lambda=0.0, end_lambda=15.0, warm_up_steps=10):
        super().__init__()
        self.name = "Loss Scheduling"
        self.start_lambda = start_lambda
        self.end_lambda = end_lambda
        self.warm_up_steps = warm_up_steps
        self.current_step = 0

    def handle_mask(self, feat, feat0, mask, lambda_mask=10, **kwargs):
        if mask is None:
            return 0.0, feat
        
        # Linear scheduling of lambda
        alpha = min(1.0, self.current_step / self.warm_up_steps)
        curr_lambda = self.start_lambda + alpha * (self.end_lambda - self.start_lambda)
        self.current_step += 1
        
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).to(feat.device).float()
            
        mask_usq = mask.unsqueeze(0).unsqueeze(0)
        if mask_usq.shape[-2:] != feat.shape[-2:]:
            mask_usq = F.interpolate(mask_usq, feat.shape[-2:], mode='bilinear')
        
        # User paints flexible area (mask=1), we fix background (mask=0)
        fixed_mask = 1.0 - mask_usq
        loss_fix = F.l1_loss(feat * fixed_mask, feat0 * fixed_mask)
        return curr_lambda * loss_fix, feat

    def reset(self):
        self.current_step = 0

def get_mask_handler(name):
    """Factory function for mask handlers."""
    if name == 'Loss Scheduling':
        return LossSchedulingMaskHandler()
    elif name == 'Baseline' or name == 'DragGAN (Baseline)':
        return BaselineMaskHandler()
    else:
        return BaselineMaskHandler()
