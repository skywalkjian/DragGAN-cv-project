import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
import numpy as np

class Tracker(ABC):
    """Base class for feature-based point tracking."""
    def __init__(self):
        self.feat_refs = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tracking_feature_idx = 5
        self.supervision_feature_idx = 5
        self.step_counter = 0

    @abstractmethod
    def track(self, feat, points, r2, h, w, **kwargs):
        """Update point positions based on feature maps."""
        pass

    def get_supervision_config(self, point_idx, r1):
        """Return motion supervision parameters."""
        return r1, 1.0

    def compute_supervision_loss(self, feat, points, targets, **kwargs):
        """Compute additional supervision loss."""
        return 0.0

    def reset(self):
        """Reset tracker internal state."""
        self.feat_refs = None
        self.step_counter = 0
        torch.cuda.empty_cache()

class BaselineTracker(Tracker):
    """Nearest Neighbor Tracking baseline."""
    def __init__(self, tracking_idx=5, supervision_idx=5):
        super().__init__()
        self.tracking_feature_idx = tracking_idx
        self.supervision_feature_idx = supervision_idx

    def track(self, feat, points, r2, h, w, **kwargs):
        feat_resize = feat[min(self.tracking_feature_idx, len(feat)-1)]

        if self.feat_refs is None:
            self.feat_refs = []
            for point in points:
                py, px = round(point[0]), round(point[1])
                py = max(0, min(h-1, py))
                px = max(0, min(w-1, px))
                self.feat_refs.append(feat_resize[0, :, py, px].detach())

        with torch.no_grad():
            for j, point in enumerate(points):
                r = round(r2)
                cy, cx = max(0, min(h-1, round(point[0]))), max(0, min(w-1, round(point[1])))
                up, down = max(cy - r, 0), min(cy + r + 1, h)
                left, right = max(cx - r, 0), min(cx + r + 1, w)
                
                feat_patch = feat_resize[:, :, up:down, left:right]
                if feat_patch.numel() == 0:
                    new_point = [cy, cx]
                else:
                    L2 = torch.linalg.norm(feat_patch - self.feat_refs[j].reshape(1, -1, 1, 1), dim=1)
                    _, idx_min = torch.min(L2.reshape(1, -1), -1)
                    
                    width = right - left
                    new_point = [idx_min.item() // width + up, idx_min.item() % width + left]
                points[j] = new_point
        
        self.step_counter += 1
        return points

class WCATTracker(Tracker):
    """Weighted Context-Aware Tracker (WCAT)."""
    def __init__(self, kernel_size=3, layer_idx=[3, 4, 5], layer_weights=[0.1, 0.4, 1.5]):
        super().__init__()
        self.kernel_size = kernel_size
        self.layer_idx = layer_idx 
        weights = torch.as_tensor(layer_weights, dtype=torch.float32)
        self.layer_weights = (weights / weights.sum()).tolist()
        self.feat_refs = None
        self.ones_kernel = None

    def track(self, feat, points, r2, h, w, **kwargs):
        # Feature fusion across multiple layers
        if isinstance(feat, list):
            f_list = []
            for i, l_idx in enumerate(self.layer_idx):
                f = feat[min(l_idx, len(feat)-1)]
                f = f * self.layer_weights[i]
                f_list.append(f)
            feat_resize = torch.cat(f_list, dim=1)
        else:
            feat_resize = feat

        C = feat_resize.shape[1]
        pad = self.kernel_size // 2

        if self.feat_refs is None:
            self.feat_refs = []
            feat_padded = F.pad(feat_resize, (pad, pad, pad, pad), mode='replicate')
            for point in points:
                py, px = round(point[0]), round(point[1])
                py_pad, px_pad = py + pad, px + pad
                y1, y2 = py_pad - pad, py_pad + pad + 1
                x1, x2 = px_pad - pad, px_pad + pad + 1
                kernel = feat_padded[0, :, y1:y2, x1:x2].detach()
                if kernel.shape[1] != self.kernel_size or kernel.shape[2] != self.kernel_size:
                    ph, pw = self.kernel_size - kernel.shape[1], self.kernel_size - kernel.shape[2]
                    kernel = F.pad(kernel, (0, pw, 0, ph), mode='replicate')
                self.feat_refs.append(kernel)
            self.ones_kernel = torch.ones(1, C, self.kernel_size, self.kernel_size, 
                                          device=feat_resize.device, dtype=feat_resize.dtype)

        with torch.no_grad():
            for j, point in enumerate(points):
                r = round(r2)
                cy, cx = max(0, min(h-1, round(point[0]))), max(0, min(w-1, round(point[1])))
                up, down = max(cy - r, 0), min(cy + r + 1, h)
                left, right = max(cx - r, 0), min(cx + r + 1, w)
                search_patch = feat_resize[:, :, up:down, left:right].contiguous()
                
                if self.ones_kernel.dtype != search_patch.dtype:
                    self.ones_kernel = self.ones_kernel.to(search_patch.dtype)
                if search_patch.shape[2] < self.kernel_size or search_patch.shape[3] < self.kernel_size:
                    continue
                
                kernel = self.feat_refs[j].unsqueeze(0).to(search_patch.dtype)
                kernel_norm_sq = torch.sum(kernel**2)
                patch_sq = search_patch**2
                patch_norm_sq = F.conv2d(patch_sq, self.ones_kernel, stride=1, padding=0)
                cross_corr = F.conv2d(search_patch, kernel, stride=1, padding=0)
                l2_sq = patch_norm_sq + kernel_norm_sq - 2 * cross_corr
                
                _, idx_min = torch.min(l2_sq.reshape(1, -1), -1)
                width = l2_sq.shape[3]
                new_point = [idx_min.item() // width + up + pad, idx_min.item() % width + left + pad]
                points[j] = new_point
        
        self.step_counter += 1
        return points

def get_tracker(name):
    """Factory function for tracker instances."""
    if name == 'Baseline':
        return BaselineTracker()
    elif name == 'WCAT':
        return WCATTracker()
    else:
        return BaselineTracker()
