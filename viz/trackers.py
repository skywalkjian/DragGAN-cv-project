import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
import numpy as np

class Tracker(ABC):
    def __init__(self):
        self.feat_refs = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_multiscale_supervision = False
        self.tracking_feature_idx = 5
        self.supervision_feature_idx = 5
        self.step_counter = 0

    @abstractmethod
    def track(self, feat, points, r2, h, w, **kwargs):
        """
        feat: Can be a list of feature maps.
        points: List of [y, x] coordinates.
        """
        pass

    def get_supervision_config(self, point_idx, r1):
        return r1, 1.0

    def compute_supervision_loss(self, feat, points, targets, **kwargs):
        """
        Optional extra loss for motion supervision.
        Can be used for multi-layer supervision or other regularization.
        """
        return 0.0

    def _get_texture_score(self, feat_res, point, window_size=5):
        """
        Helper to calculate texture richness around a point.
        Uses local variance across channels.
        """
        h, w = feat_res.shape[2], feat_res.shape[3]
        py, px = int(round(point[0])), int(round(point[1]))
        
        r = window_size // 2
        up = max(py - r, 0)
        down = min(py + r + 1, h)
        left = max(px - r, 0)
        right = min(px + r + 1, w)
        
        patch = feat_res[0, :, up:down, left:right]
        if patch.numel() == 0:
            return 1.0
            
        # Variance as texture score
        var = torch.var(patch, dim=(1, 2)).mean().item()
        return var

    def reset(self):
        self.feat_refs = None
        self.step_counter = 0
        torch.cuda.empty_cache()

class DragGANTracker(Tracker):
    """Baseline Nearest Neighbor Tracking."""
    def __init__(self, tracking_idx=5, supervision_idx=5, dynamic_switch=False):
        super().__init__()
        self.tracking_feature_idx = tracking_idx
        self.supervision_feature_idx = supervision_idx
        self.dynamic_switch = dynamic_switch

    def track(self, feat, points, r2, h, w, **kwargs):
        # Determine current tracking index
        idx = self.tracking_feature_idx
        if self.dynamic_switch:
            # If dynamic_switch is True, first step use Layer 5, then switch to the target index
            if self.step_counter == 0:
                idx = 5
            else:
                pass
        
        feat_resize = feat[min(idx, len(feat)-1)]

        if self.feat_refs is None:
            self.feat_refs = []
            for point in points:
                py, px = round(point[0]), round(point[1])
                py = max(0, min(h-1, py))
                px = max(0, min(w-1, px))
                self.feat_refs.append(feat_resize[0, :, py, px].detach())
        
        # In dynamic mode, we need to update feat_refs if we just switched layers
        if self.dynamic_switch and self.step_counter == 1:
            self.feat_refs = []
            for point in points:
                py, px = round(point[0]), round(point[1])
                py = max(0, min(h-1, py))
                px = max(0, min(w-1, px))
                self.feat_refs.append(feat_resize[0, :, py, px].detach())

        with torch.no_grad():
            for j, point in enumerate(points):
                r = round(r2)
                # Ensure pred coordinates are within bounds
                cy, cx = max(0, min(h-1, round(point[0]))), max(0, min(w-1, round(point[1])))
                up = max(cy - r, 0)
                down = min(cy + r + 1, h)
                left = max(cx - r, 0)
                right = min(cx + r + 1, w)
                
                feat_patch = feat_resize[:, :, up:down, left:right]
                if feat_patch.numel() == 0:
                    new_point = [cy, cx] # Fallback to current point if patch is empty
                else:
                    L2 = torch.linalg.norm(feat_patch - self.feat_refs[j].reshape(1, -1, 1, 1), dim=1)
                    _, idx_min = torch.min(L2.reshape(1, -1), -1)
                    
                    width = right - left
                    new_point = [idx_min.item() // width + up, idx_min.item() % width + left]
                points[j] = new_point
        
        self.step_counter += 1
        return points

    def reset(self):
        super().reset()
        self.step_counter = 0

class RAFTTracker(Tracker):
    """
    Optical Flow Tracking based on RAFT (Recurrent All-Pairs Field Transforms).
    This tracker uses the motion between consecutive generated frames to update control points.
    It provides robust tracking even when the image undergoes significant transformations.
    """
    def __init__(self):
        super().__init__()
        # Load pre-trained RAFT model from torchvision
        self.model = raft_large(weights=Raft_Large_Weights.DEFAULT).to(self.device).eval()
        self.prev_img = None

    def track(self, feat, points, r2, h, w, **kwargs):
        """
        Updates point positions using optical flow between previous and current frames.
        """
        curr_img = kwargs.get('curr_img') # [1, 3, H, W]
        if self.prev_img is None or curr_img is None:
            self.prev_img = curr_img.detach() if curr_img is not None else None
            return points

        with torch.no_grad():
            # Prepare images for RAFT: rescale to [0, 255]
            img1 = (self.prev_img * 127.5 + 128).clamp(0, 255).float()
            img2 = (curr_img * 127.5 + 128).clamp(0, 255).float()
            
            # Predict optical flow
            list_predictions = self.model(img1, img2)
            flow = list_predictions[-1] # [1, 2, H, W]
            
            new_points = []
            for j, point in enumerate(points):
                # Sample flow at point location using bilinear interpolation for sub-pixel precision
                grid_y = (point[0] / (h - 1)) * 2 - 1
                grid_x = (point[1] / (w - 1)) * 2 - 1
                grid = torch.tensor([[[[grid_x, grid_y]]]], device=self.device).float()
                
                # Sample flow field: flow is [1, 2, H, W], grid is [1, 1, 1, 2]
                sampled_flow = F.grid_sample(flow, grid, mode='bilinear', align_corners=True)
                
                dx = sampled_flow[0, 0, 0, 0].item()
                dy = sampled_flow[0, 1, 0, 0].item()
                
                new_points.append([point[0] + dy, point[1] + dx])
            
            self.prev_img = curr_img.detach()
            return new_points

    def reset(self):
        super().reset()
        self.prev_img = None

class DeepParticleTracker(Tracker):
    """
    Advanced Particle-based tracker implementing iterative refinement and 
    multi-scale patch correlation, inspired by the PIPs architecture.
    """
    def __init__(self, tracking_layers=[3, 5, 7], patch_size=8, iterations=3, momentum=0.8):
        super().__init__()
        self.tracking_layers = tracking_layers
        self.patch_size = patch_size
        self.iterations = iterations
        self.momentum = momentum
        
        self.feat_refs = None # Appearance memory
        self.velocity = None  # Motion memory
        self.history = None   # Trajectory history
        
    def _extract_patches(self, feats, points, h, w):
        """Extracts local patches across multiple feature layers for correlation."""
        patches_all_layers = {}
        pad = self.patch_size // 2
        
        for layer_idx in self.tracking_layers:
            feat = feats[min(layer_idx, len(feats)-1)]
            feat = F.normalize(feat, p=2, dim=1) # Normalize for cosine similarity
            
            feat_padded = F.pad(feat, (pad, pad, pad, pad), mode='replicate')
            
            layer_patches = []
            for p in points:
                py, px = int(round(p[0])) + pad, int(round(p[1])) + pad
                patch = feat_padded[0, :, py-pad:py+pad, px-pad:px+pad].detach()
                layer_patches.append(patch)
            patches_all_layers[layer_idx] = layer_patches
            
        return patches_all_layers

    def track(self, feat, points, r2, h, w, **kwargs):
        if not isinstance(feat, list):
            feat = [feat]
            
        if self.feat_refs is None:
            self.feat_refs = self._extract_patches(feat, points, h, w)
            self.velocity = [[0.0, 0.0] for _ in points]
            self.history = [[p[:]] for p in points]
            return points

        curr_feats = {}
        for layer_idx in self.tracking_layers:
            f = feat[min(layer_idx, len(feat)-1)]
            curr_feats[layer_idx] = F.normalize(f, p=2, dim=1)

        new_points = []
        with torch.no_grad():
            for j, p in enumerate(points):
                # Prediction step: Motion Prior
                curr_y = p[0] + self.velocity[j][0] * self.momentum
                curr_x = p[1] + self.velocity[j][1] * self.momentum
                
                # Iterative Refinement via Local Cost Volume
                for _ in range(self.iterations):
                    total_score = None
                    r = round(r2)
                    
                    up, down = max(int(round(curr_y)) - r, 0), min(int(round(curr_y)) + r + 1, h)
                    left, right = max(int(round(curr_x)) - r, 0), min(int(round(curr_x)) + r + 1, w)
                    
                    if (down - up) < self.patch_size or (right - left) < self.patch_size:
                        continue
                    
                    for layer_idx in self.tracking_layers:
                        f_map = curr_feats[layer_idx][:, :, up:down, left:right]
                        ref_patch = self.feat_refs[layer_idx][j].unsqueeze(0)
                        
                        # Cross-Correlation to find the best matching position
                        score = F.conv2d(f_map, ref_patch.to(f_map.dtype))
                        
                        if total_score is None:
                            total_score = score
                        else:
                            total_score += score
                    
                    if total_score is not None:
                        _, idx = torch.max(total_score.reshape(1, -1), -1)
                        out_h, out_w = total_score.shape[2], total_score.shape[3]
                        
                        dy_local = idx.item() // out_w
                        dx_local = idx.item() % out_w
                        
                        # Sub-pixel refinement using center-of-mass within a 3x3 window
                        if 0 < dy_local < out_h - 1 and 0 < dx_local < out_w - 1:
                            patch_3x3 = total_score[0, 0, dy_local-1:dy_local+2, dx_local-1:dx_local+2]
                            weights = F.softmax(patch_3x3.reshape(-1) * 10.0, dim=0).reshape(3, 3)
                            
                            yy, xx = torch.meshgrid(torch.arange(-1, 2), torch.arange(-1, 2), indexing='ij')
                            yy = yy.to(total_score.device).to(total_score.dtype)
                            xx = xx.to(total_score.device).to(total_score.dtype)
                            
                            dy_local += torch.sum(yy * weights).item()
                            dx_local += torch.sum(xx * weights).item()

                        curr_y = up + dy_local + self.patch_size // 2
                        curr_x = left + dx_local + self.patch_size // 2
                
                new_points.append([curr_y, curr_x])
                self.velocity[j] = [curr_y - points[j][0], curr_x - points[j][1]]
                self.history[j].append([curr_y, curr_x])
        
        torch.cuda.empty_cache()
        self.step_counter += 1
        return new_points

    def reset(self):
        super().reset()
        self.feat_refs = None
        self.velocity = None
        self.history = None

class ContextAwareTracker(Tracker):
    """
    Weighted Context-Aware Tracker (WCAT).
    Matches spatial context patches instead of single points to enforce structural consistency.
    Supports multi-layer feature fusion for improved localization.
    """
    def __init__(self, kernel_size=3, layer_idx=5, layer_weights=None, stride=1, dilation=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.layer_idx = layer_idx 
        
        if layer_weights is not None:
            weights = torch.as_tensor(layer_weights, dtype=torch.float32)
            self.layer_weights = (weights / weights.sum()).tolist()
        else:
            self.layer_weights = None
            
        self.stride = stride
        self.dilation = dilation
        self.feat_refs = None
        self.ones_kernel = None

    def track(self, feat, points, r2, h, w, **kwargs):
        """
        Tracks points by matching local context patches across specified feature layers.
        """
        # Multi-layer feature fusion
        if isinstance(feat, list):
            if isinstance(self.layer_idx, list):
                # Fuse multiple layers
                f_list = []
                for i, l_idx in enumerate(self.layer_idx):
                    f = feat[min(l_idx, len(feat)-1)]
                    if self.layer_weights is not None:
                        f = f * self.layer_weights[i]
                    f_list.append(f)
                feat_resize = torch.cat(f_list, dim=1)
            else:
                feat_resize = feat[min(self.layer_idx, len(feat)-1)]
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
            self.ones_kernel = torch.ones(1, C, self.kernel_size, self.kernel_size, device=feat_resize.device, dtype=feat_resize.dtype)

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
                
                # Context-aware patch matching via convolution
                # search_patch: [1, C, H_p, W_p], kernel: [C, 1, k, k]
                kernel = self.feat_refs[j].unsqueeze(0).to(search_patch.dtype)
                
                # L2 distance via convolution: ||A-B||^2 = ||A||^2 + ||B||^2 - 2<A,B>
                # 1. ||B||^2 (kernel norm)
                kernel_norm_sq = torch.sum(kernel**2)
                
                # 2. ||A||^2 (local patch norms)
                patch_sq = search_patch**2
                patch_norm_sq = F.conv2d(patch_sq, self.ones_kernel, stride=self.stride, padding=0)
                
                # 3. <A,B> (cross-correlation)
                cross_corr = F.conv2d(search_patch, kernel, stride=self.stride, padding=0)
                
                # 4. Combine
                l2_sq = patch_norm_sq + kernel_norm_sq - 2 * cross_corr
                
                # Find minimum
                _, idx_min = torch.min(l2_sq.reshape(1, -1), -1)
                
                width = l2_sq.shape[3]
                new_point = [idx_min.item() // width + up + pad, idx_min.item() % width + left + pad]
                points[j] = new_point
        
        self.step_counter += 1
        return points

def get_tracker(name):
    """
    Factory function to return the specified tracker instance.
    English names are used for consistency with the UI.
    """
    if name == 'DragGAN (Baseline)':
        return DragGANTracker(tracking_idx=5, supervision_idx=5)
    elif name == 'RAFT Large (Optical Flow)':
        return RAFTTracker()
    elif name == 'Deep Particle (PIPs-like)':
        return DeepParticleTracker()
    elif name == 'WCAT (Weighted Context-Aware Tracker)':
        return ContextAwareTracker(kernel_size=3, layer_idx=[3, 4, 5], layer_weights=[0.1, 0.4, 1.5])
    else:
        # Default to baseline
        return DragGANTracker()
