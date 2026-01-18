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
                print(f"Step 0: Using Layer 5 for initial reference and first track")
            else:
                print(f"Step {self.step_counter}: Switched to Layer {idx}")
        
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
            print(f"Dynamic Switch: Re-extracted references for Layer {idx}")

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
    Optical Flow Tracking using RAFT (Large).
    Uses the motion between consecutive frames to update point positions.
    """
    def __init__(self):
        super().__init__()
        self.model = raft_large(weights=Raft_Large_Weights.DEFAULT).to(self.device).eval()
        self.prev_img = None

    def track(self, feat, points, r2, h, w, **kwargs):
        curr_img = kwargs.get('curr_img') # [1, 3, H, W]
        if self.prev_img is None or curr_img is None:
            self.prev_img = curr_img.detach() if curr_img is not None else None
            return points

        with torch.no_grad():
            # RAFT expects [0, 255] range and specific normalization
            img1 = (self.prev_img * 127.5 + 128).clamp(0, 255).float()
            img2 = (curr_img * 127.5 + 128).clamp(0, 255).float()
            
            # RAFT Large might need more memory, ensure it fits
            list_predictions = self.model(img1, img2)
            flow = list_predictions[-1] # [1, 2, H, W]
            
            new_points = []
            for j, point in enumerate(points):
                # Bilinear sampling from flow field for sub-pixel precision
                grid_y = (point[0] / (h - 1)) * 2 - 1
                grid_x = (point[1] / (w - 1)) * 2 - 1
                # grid must be 4D: (N, H_out, W_out, 2)
                grid = torch.tensor([[[[grid_x, grid_y]]]], device=self.device).float()
                
                # flow is [1, 2, H, W], grid is [1, 1, 1, 2]
                # sampled_flow will be [1, 2, 1, 1]
                sampled_flow = F.grid_sample(flow, grid, mode='bilinear', align_corners=True)
                
                dx = sampled_flow[0, 0, 0, 0].item()
                dy = sampled_flow[0, 1, 0, 0].item()
                
                new_points.append([point[0] + dy, point[1] + dx])
            
            self.prev_img = curr_img.detach()
            return new_points

    def reset(self):
        super().reset()
        self.prev_img = None

class PIPsInspiredTracker(Tracker):
    """
    PIPs-Inspired Long-term Tracker.
    Features:
    1. Temporal Momentum: Prevents sudden jumps in low-texture areas by smoothing the trajectory.
    2. Dynamic Feature Memory: Maintains a moving average of the tracked feature to adapt to appearance changes.
    3. Multi-scale Coarse-to-Fine: Refines position across multiple feature layers.
    """
    def __init__(self, tracking_idx=3, momentum=0.7, feat_alpha=0.0):
        super().__init__()
        self.tracking_feature_idx = tracking_idx
        self.momentum = momentum # Weight for velocity/momentum
        self.feat_alpha = feat_alpha # Update rate for feature memory (0.0 = no update, prevents drift)
        self.prev_points = None
        self.velocity = None # Momentum term
        self.feat_refs = None # Dynamic feature templates

    def track(self, feat, points, r2, h, w, **kwargs):
        # Use Layer 3 for robustness, but we can also check Layer 5 for precision
        idx_robust = self.tracking_feature_idx
        idx_precise = 5
        
        feat_robust = feat[min(idx_robust, len(feat)-1)]
        feat_precise = feat[min(idx_precise, len(feat)-1)]

        if self.feat_refs is None:
            self.feat_refs = []
            for point in points:
                py, px = round(point[0]), round(point[1])
                py, px = max(0, min(h-1, py)), max(0, min(w-1, px))
                # Store features from both layers
                f_r = feat_robust[0, :, py, px].detach()
                f_p = feat_precise[0, :, py, px].detach()
                self.feat_refs.append({'robust': f_r, 'precise': f_p})
            
            self.prev_points = [p[:] for p in points]
            self.velocity = [[0.0, 0.0] for _ in points]
            return points

        new_points = []
        with torch.no_grad():
            for j, point in enumerate(points):
                # 1. Prediction step: Apply momentum
                pred_y = point[0] + self.velocity[j][0] * self.momentum
                pred_x = point[1] + self.velocity[j][1] * self.momentum
                
                # 2. Robust Search (Layer 3/Robust)
                r = round(r2)
                # Ensure pred coordinates are within bounds
                cy, cx = max(0, min(h-1, round(pred_y))), max(0, min(w-1, round(pred_x)))
                up = max(cy - r, 0)
                down = min(cy + r + 1, h)
                left = max(cx - r, 0)
                right = min(cx + r + 1, w)
                
                patch_r = feat_robust[:, :, up:down, left:right]
                if patch_r.numel() == 0:
                    p_r = [cy, cx]
                else:
                    L2_r = torch.linalg.norm(patch_r - self.feat_refs[j]['robust'].reshape(1, -1, 1, 1), dim=1)
                    _, min_idx_r = torch.min(L2_r.reshape(1, -1), -1)
                    
                    width = right - left
                    p_r = [min_idx_r.item() // width + up, min_idx_r.item() % width + left]
                
                # 3. Precise Refinement (Layer 5/Precise) - Search in a smaller radius
                r_small = max(2, r // 2)
                cy_r, cx_r = max(0, min(h-1, round(p_r[0]))), max(0, min(w-1, round(p_r[1])))
                up_s = max(cy_r - r_small, 0)
                down_s = min(cy_r + r_small + 1, h)
                left_s = max(cx_r - r_small, 0)
                right_s = min(cx_r + r_small + 1, w)
                
                patch_p = feat_precise[:, :, up_s:down_s, left_s:right_s]
                if patch_p.numel() == 0:
                    p_final = [cy_r, cx_r]
                else:
                    L2_p = torch.linalg.norm(patch_p - self.feat_refs[j]['precise'].reshape(1, -1, 1, 1), dim=1)
                    _, min_idx_p = torch.min(L2_p.reshape(1, -1), -1)
                    
                    width_s = right_s - left_s
                    p_final = [min_idx_p.item() // width_s + up_s, min_idx_p.item() % width_s + left_s]
                
                # 4. Update state
                # Calculate new velocity
                self.velocity[j] = [p_final[0] - self.prev_points[j][0], p_final[1] - self.prev_points[j][1]]
                self.prev_points[j] = p_final[:]
                
                # Dynamic feature update (moving average)
                f_r_curr = feat_robust[0, :, int(round(p_final[0])), int(round(p_final[1]))].detach()
                f_p_curr = feat_precise[0, :, int(round(p_final[0])), int(round(p_final[1]))].detach()
                
                self.feat_refs[j]['robust'] = (1 - self.feat_alpha) * self.feat_refs[j]['robust'] + self.feat_alpha * f_r_curr
                self.feat_refs[j]['precise'] = (1 - self.feat_alpha) * self.feat_refs[j]['precise'] + self.feat_alpha * f_p_curr
                
                new_points.append(p_final)
                
        self.step_counter += 1
        return new_points

    def reset(self):
        super().reset()
        self.prev_points = None
        self.velocity = None
        self.feat_refs = None

class DeepParticleTracker(Tracker):
    """
    A more complete particle-based tracker inspired by PIPs (Particle Video Tracking).
    
    Key Features:
    1. Multi-scale Patch Correlation: Uses local patches from multiple feature layers.
    2. Iterative Refinement: Iteratively updates position using a local cost volume.
    3. Motion & Appearance Memory: Maintains a hidden state to smooth trajectory and adapt to changes.
    4. Sub-pixel Precision: Uses bilinear interpolation for patch extraction and coordinate updates.
    """
    def __init__(self, tracking_layers=[3, 5, 7], patch_size=8, iterations=3, momentum=0.8):
        super().__init__()
        self.tracking_layers = tracking_layers
        self.patch_size = patch_size
        self.iterations = iterations
        self.momentum = momentum
        
        self.feat_refs = None # Appearance memory: {layer_idx: [points_patches]}
        self.velocity = None  # Motion memory: [[dy, dx], ...]
        self.history = None   # Latent trajectory history
        
    def _extract_patches(self, feats, points, h, w):
        """Extract patches for each point across all tracking layers."""
        patches_all_layers = {}
        pad = self.patch_size // 2
        
        for layer_idx in self.tracking_layers:
            feat = feats[min(layer_idx, len(feats)-1)]
            # Normalize for correlation
            feat = F.normalize(feat, p=2, dim=1)
            feat_res = feat
            C = feat_res.shape[1]
            
            # Pad to handle borders
            feat_padded = F.pad(feat_res, (pad, pad, pad, pad), mode='replicate')
            
            layer_patches = []
            for p in points:
                py, px = int(round(p[0])) + pad, int(round(p[1])) + pad
                # Extract KxK patch
                patch = feat_padded[0, :, py-pad:py+pad, px-pad:px+pad].detach() # (C, K, K)
                layer_patches.append(patch)
            patches_all_layers[layer_idx] = layer_patches
            
        return patches_all_layers

    def track(self, feat, points, r2, h, w, **kwargs):
        if not isinstance(feat, list):
            feat = [feat] # Ensure it's a list
            
        # 1. Initialize memory if first step
        if self.feat_refs is None:
            self.feat_refs = self._extract_patches(feat, points, h, w)
            self.velocity = [[0.0, 0.0] for _ in points]
            self.history = [[p[:]] for p in points]
            return points

        # 2. Prepare features for current frame
        curr_feats = {}
        for layer_idx in self.tracking_layers:
            f = feat[min(layer_idx, len(feat)-1)]
            f = F.normalize(f, p=2, dim=1)
            # No resizing needed here (handled by renderer)
            curr_feats[layer_idx] = f

        new_points = []
        with torch.no_grad():
            for j, p in enumerate(points):
                # Start from predicted position (Motion Prior)
                curr_y = p[0] + self.velocity[j][0] * self.momentum
                curr_x = p[1] + self.velocity[j][1] * self.momentum
                
                # Iterative Refinement (PIPs-style)
                for _ in range(self.iterations):
                    # Compute total cost map from all layers
                    total_score = None
                    r = round(r2)
                    
                    # Define search window around current estimate
                    up, down = max(int(round(curr_y)) - r, 0), min(int(round(curr_y)) + r + 1, h)
                    left, right = max(int(round(curr_x)) - r, 0), min(int(round(curr_x)) + r + 1, w)
                    
                    if (down - up) < self.patch_size or (right - left) < self.patch_size:
                        continue # Skip refinement if window too small
                    
                    for layer_idx in self.tracking_layers:
                        f_map = curr_feats[layer_idx][:, :, up:down, left:right]
                        ref_patch = self.feat_refs[layer_idx][j].unsqueeze(0) # (1, C, K, K)
                        
                        # Ensure dtype consistency
                        if ref_patch.dtype != f_map.dtype:
                            ref_patch = ref_patch.to(f_map.dtype)
                            
                        # Use Cross-Correlation (higher is better)
                        # Result shape: (1, 1, H_out, W_out)
                        score = F.conv2d(f_map, ref_patch)
                        
                        if total_score is None:
                            total_score = score
                        else:
                            # Accumulate scores across layers
                            total_score += score
                    
                    if total_score is not None:
                        # Find max correlation (best match)
                        # For sub-pixel accuracy, we could use a soft-argmax or quadratic fitting,
                        # but for now, we'll implement a simple sub-pixel refinement if possible.
                        _, idx = torch.max(total_score.reshape(1, -1), -1)
                        out_h, out_w = total_score.shape[2], total_score.shape[3]
                        
                        # Local offset in score map
                        dy_local = idx.item() // out_w
                        dx_local = idx.item() % out_w
                        
                        # Sub-pixel refinement via center of mass (soft-argmax like)
                        # We take a 3x3 window around the max and compute weighted average
                        if 0 < dy_local < out_h - 1 and 0 < dx_local < out_w - 1:
                            patch_3x3 = total_score[0, 0, dy_local-1:dy_local+2, dx_local-1:dx_local+2]
                            # Softmax to get weights - use reshape instead of view to handle non-contiguous patches
                            weights = F.softmax(patch_3x3.reshape(-1) * 10.0, dim=0).reshape(3, 3)
                            
                            yy, xx = torch.meshgrid(torch.arange(-1, 2), torch.arange(-1, 2), indexing='ij')
                            yy = yy.to(total_score.device).to(total_score.dtype)
                            xx = xx.to(total_score.device).to(total_score.dtype)
                            
                            sub_dy = torch.sum(yy * weights).item()
                            sub_dx = torch.sum(xx * weights).item()
                            
                            dy_local += sub_dy
                            dx_local += sub_dx

                        # Update current estimate (Map back to global)
                        # Center of the best matching patch
                        curr_y = up + dy_local + self.patch_size // 2
                        curr_x = left + dx_local + self.patch_size // 2
                
                # Final position for this point
                new_points.append([curr_y, curr_x])
                
                # Update velocity
                self.velocity[j] = [curr_y - points[j][0], curr_x - points[j][1]]
                self.history[j].append([curr_y, curr_x])
        
        # Free memory
        del curr_feats
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
    Context-Aware Tracker inspired by Context-PIPs.
    Instead of matching a single point, it matches a spatial context patch (Kernel)
    to enforce local structural consistency.
    Supports single layer or multi-layer fusion (concatenation).
    """
    def __init__(self, kernel_size=3, layer_idx=5, layer_weights=None, stride=1, dilation=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.layer_idx = layer_idx  # Can be int or list of ints
        
        # Normalize weights if provided
        if layer_weights is not None:
            weights = torch.as_tensor(layer_weights, dtype=torch.float32)
            self.layer_weights = (weights / weights.sum()).tolist()
        else:
            self.layer_weights = None
            
        self.stride = stride
        self.dilation = dilation
        self.feat_refs = None
        # Pre-compute ones kernel for local energy calculation
        self.ones_kernel = None

    def track(self, feat, points, r2, h, w, **kwargs):
        if isinstance(feat, list):
            if isinstance(self.layer_idx, list):
                # Feature Fusion: Concatenate normalized features from multiple layers
                feats_to_fuse = []
                for i, idx in enumerate(self.layer_idx):
                    f = feat[min(idx, len(feat)-1)]
                    f = F.normalize(f, p=2, dim=1)
                    # Apply layer weight if provided
                    if self.layer_weights is not None:
                        f = f * self.layer_weights[i]
                    feats_to_fuse.append(f)
                feat_resize = torch.cat(feats_to_fuse, dim=1)
                feat_resize = F.normalize(feat_resize, p=2, dim=1)
            else:
                idx = self.layer_idx
                feat_resize = feat[min(idx, len(feat)-1)]
                feat_resize = F.normalize(feat_resize, p=2, dim=1)
        else:
            feat_resize = feat
            feat_resize = F.normalize(feat_resize, p=2, dim=1)

        C = feat_resize.shape[1]
        pad = self.kernel_size // 2

        # Initialize reference templates (kernels)
        if self.feat_refs is None:
            self.feat_refs = []
            # Pad feature map to handle borders
            feat_padded = F.pad(feat_resize, (pad, pad, pad, pad), mode='replicate')
            
            for point in points:
                py = int(torch.round(torch.as_tensor(point[0])).item())
                px = int(torch.round(torch.as_tensor(point[1])).item())
                py = max(0, min(h - 1, py))
                px = max(0, min(w - 1, px))
                
                py_pad, px_pad = py + pad, px + pad
                y1, y2 = py_pad - pad, py_pad + pad + 1
                x1, x2 = px_pad - pad, px_pad + pad + 1
                
                kernel = feat_padded[0, :, y1:y2, x1:x2].detach()
                if kernel.shape[1] != self.kernel_size or kernel.shape[2] != self.kernel_size:
                    ph = self.kernel_size - kernel.shape[1]
                    pw = self.kernel_size - kernel.shape[2]
                    kernel = F.pad(kernel, (0, pw, 0, ph), mode='replicate')
                
                self.feat_refs.append(kernel)
            
            # Initialize ones kernel for energy calculation
            self.ones_kernel = torch.ones(1, C, self.kernel_size, self.kernel_size, device=feat_resize.device, dtype=feat_resize.dtype)

        with torch.no_grad():
            for j, point in enumerate(points):
                r = round(r2)
                
                # Current point
                cy = max(0, min(h-1, round(point[0])))
                cx = max(0, min(w-1, round(point[1])))
                
                # Define search window
                up, down = max(cy - r, 0), min(cy + r + 1, h)
                left, right = max(cx - r, 0), min(cx + r + 1, w)
                
                # Extract search patch
                search_patch = feat_resize[:, :, up:down, left:right].contiguous()
                
                if self.ones_kernel.dtype != search_patch.dtype:
                    self.ones_kernel = self.ones_kernel.to(search_patch.dtype)
                
                if search_patch.shape[2] < self.kernel_size or search_patch.shape[3] < self.kernel_size:
                    continue

                # Template Matching via Convolution
                ref_kernel = self.feat_refs[j].unsqueeze(0).contiguous()
                term1 = F.conv2d(search_patch, ref_kernel)
                term2 = F.conv2d((search_patch ** 2).contiguous(), self.ones_kernel)
                
                # L2 Squared Distance map
                score_map = term2 - 2 * term1
                score_map = score_map[0, 0]
                
                # Spatial penalty
                sy, sx = score_map.shape[0], score_map.shape[1]
                grid_y, grid_x = torch.meshgrid(torch.arange(sy, device=score_map.device), torch.arange(sx, device=score_map.device), indexing='ij')
                
                target_dy, target_dx = cy - up - pad, cx - left - pad
                dist_sq = (grid_y - target_dy)**2 + (grid_x - target_dx)**2
                score_map = score_map + 0.01 * dist_sq
                
                # Find minimum
                min_val = torch.min(score_map)
                min_idx = torch.nonzero(score_map == min_val, as_tuple=False)[0] 
                dy, dx = min_idx[0].item(), min_idx[1].item()
                
                # New point
                new_py = up + dy + pad
                new_px = left + dx + pad
                
                # Boundary clamping
                new_py = max(0.0, min(float(h-1), float(new_py)))
                new_px = max(0.0, min(float(w-1), float(new_px)))
                
                points[j] = [new_py, new_px]
        
        self.step_counter += 1
        return points

def get_tracker(name):
    if name == 'DragGAN (Baseline)':
        return DragGANTracker(tracking_idx=5, supervision_idx=5)
    elif name == 'RAFT Large (Optical Flow)':
        return RAFTTracker()
    elif name == 'Deep Particle (PIPs-like)':
        return DeepParticleTracker()
    elif name == 'PIPs-Inspired (Momentum)':
        return PIPsInspiredTracker(tracking_idx=3)
    elif name == 'WCAT (Weighted Context-Aware Tracker)':
        return ContextAwareTracker(kernel_size=3, layer_idx=[3, 4, 5], layer_weights=[0.1, 0.4, 1.5])
    else:
        return DragGANTracker()
