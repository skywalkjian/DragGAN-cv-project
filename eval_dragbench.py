import os
import torch
import numpy as np
import json
import pickle
import random
from PIL import Image
from tqdm import tqdm
import dnnlib
from viz.renderer import Renderer
from viz.trackers import get_tracker
from viz.mask_handlers import get_mask_handler
import argparse
import warnings
import shutil
from calculate_fid import compute_fid_from_paths

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

"""
DragBench Evaluation Script (Full Version)
Supports automated testing on DragBench structure and metrics (MD, FID).
"""

def set_random_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Info] Random seed set to {seed}")

def run_eval(config_name, test_cases, steps=50, device='cuda', mask_handler_name='Baseline', use_mask=True, save_dir=None):
    renderer = Renderer(disable_timing=True)
    results = []
    
    mask_label = mask_handler_name if use_mask else "None (Pure Tracker)"
    print(f"Starting evaluation for {config_name} (Steps: {steps}, Mask: {mask_label})")
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        # Create a reference directory for FID if it doesn't exist
        ref_dir = os.path.join(os.path.dirname(save_dir), 'reference')
        os.makedirs(ref_dir, exist_ok=True)

    for idx, case in enumerate(test_cases):
        if args_cmd.num and idx >= args_cmd.num:
            break
        
        case_name = case.get('name', f'case_{idx}')
        # print(f"  Processing case {idx}/{min(len(test_cases), args_cmd.num if args_cmd.num else 999)}: {case_name}")
        
        pkl_path = case.get('pkl', 'checkpoints/stylegan2-ffhq-512x512.pkl')
        if not os.path.exists(pkl_path):
            continue
            
        points = case['points']
        targets = case['targets']
        
        # Scale points to model resolution if source resolution is known
        if 'source_res' in case:
            target_res = renderer.get_resolution(pkl_path)
            src_h, src_w = case['source_res']
            scale_h, scale_w = target_res / src_h, target_res / src_w
            points = [[p[0] * scale_h, p[1] * scale_w] for p in points]
            targets = [[p[0] * scale_h, p[1] * scale_w] for p in targets]
        
        if len(points) == 0:
            continue
        seed = case.get('seed', 0)
        
        args = dnnlib.EasyDict(
            pkl=pkl_path,
            w0_seed=seed,
            batch_size=1,
            lr=0.001,
            w_plus=True,
            reset_w=True,
            is_drag=True,
            points=points,
            targets=targets,
            mask=case.get('mask', None) if use_mask else None,
            feature_idx=5,
            tracker_type=config_name
        )
        
        renderer.tracker = get_tracker(config_name)
        renderer.mask_handler = get_mask_handler(mask_handler_name)
        
        # Initial render to setup weights and save reference image for FID
        res0 = renderer.render(**args)
        if save_dir:
            ref_path = os.path.join(os.path.dirname(save_dir), 'reference', f"{case_name}_orig.png")
            if not os.path.exists(ref_path):
                img0 = res0.image
                if isinstance(img0, torch.Tensor):
                    img0 = Image.fromarray(img0.detach().cpu().numpy().astype(np.uint8))
                else:
                    img0 = Image.fromarray(np.array(img0).astype(np.uint8))
                
                # Enforce max 512x512 resolution for evaluation
                if img0.width > 512 or img0.height > 512:
                    img0.thumbnail((512, 512), Image.LANCZOS)
                
                img0.save(ref_path)
        
        current_points = [p[:] for p in points]
        args.reset_w = False
        
        last_md = 0
        for i in range(steps):
            args.points = current_points
            res = renderer.render(**args)
            if 'error' in res and res.error:
                # Handle CapturedException or other error types
                err_msg = str(res.error)
                print(f"  [!] Error in {case_name} step {i}: {err_msg}")
                break
            if 'md' in res:
                last_md = res.md
            if 'points' in res:
                current_points = [[p[0], p[1]] for p in res.points]
            
            # Save final result for FID and visualization
            if i == steps - 1:
                if save_dir and 'image' in res:
                    img = res.image
                    if isinstance(img, torch.Tensor):
                        img = Image.fromarray(img.detach().cpu().numpy().astype(np.uint8))
                    else:
                        img = Image.fromarray(np.array(img).astype(np.uint8))
                    
                    # Enforce max 512x512 resolution for evaluation
                    if img.width > 512 or img.height > 512:
                        img.thumbnail((512, 512), Image.LANCZOS)
                        
                    out_path = os.path.join(save_dir, f"{case_name}_final.png")
                    img.save(out_path)

        results.append({
            'name': case_name,
            'category': case.get('category', 'unknown'),
            'md': float(last_md)
        })
        
    return results

def load_dragbench(path):
    # Mapping from folder keywords to available StyleGAN checkpoints
    CATEGORY_TO_PKL = {
        'cat': 'checkpoints/stylegan2-cat-config-f.pkl',
        'dog': 'checkpoints/stylegan2_dogs_1024_pytorch.pkl',
        'horse': 'checkpoints/stylegan2_horses_256_pytorch.pkl',
        'elephant': 'checkpoints/stylegan2_elephants_512_pytorch.pkl',
        'elephants': 'checkpoints/stylegan2_elephants_512_pytorch.pkl',
        'lion': 'checkpoints/stylegan2_lions_512_pytorch.pkl',
        'human_head': 'checkpoints/stylegan2-ffhq-512x512.pkl',
        'face': 'checkpoints/stylegan2-ffhq-512x512.pkl',
        'ffhq': 'checkpoints/stylegan2-ffhq-512x512.pkl',
        'afhqcat': 'checkpoints/stylegan2-afhqcat-512x512.pkl',
        'car': 'checkpoints/stylegan2-car-config-f.pkl',
    }

    cases = []
    if not os.path.exists(path):
        return None
        
    print(f"(+) Scanning DragBench folders for matching StyleGAN models...")
    for root, dirs, files in os.walk(path):
        handle_points = []
        target_points = []
        mask_data = None
        source_res = None
        
        # Determine appropriate checkpoint based on path
        path_lower = root.lower()
        assigned_pkl = None
        matched_kw = None
        for kw, pkl_path in CATEGORY_TO_PKL.items():
            if kw in path_lower:
                if os.path.exists(pkl_path):
                    assigned_pkl = pkl_path
                    matched_kw = kw
                    break
        
        if not assigned_pkl:
            continue

        # Try to find an image to determine source resolution
        for img_name in ['origin_image.png', 'original_image.png', 'input_image.png']:
            img_path = os.path.join(root, img_name)
            if os.path.exists(img_path):
                with Image.open(img_path) as img:
                    source_res = (img.height, img.width)
                break

        if 'meta_data.json' in files:
            with open(os.path.join(root, 'meta_data.json'), 'r') as f:
                meta = json.load(f)
                handle_points = [[p[1], p[0]] for p in meta.get('handle_points', [])]
                target_points = [[p[1], p[0]] for p in meta.get('target_points', [])]
                mask_data = meta.get('mask', None)
        elif 'meta_data.pkl' in files:
            import pickle
            with open(os.path.join(root, 'meta_data.pkl'), 'rb') as f:
                meta = pickle.load(f)
                raw_points = meta.get('points', [])
                handle_points = [[p[1], p[0]] for p in raw_points[0::2]]
                target_points = [[p[1], p[0]] for p in raw_points[1::2]]
        elif 'prompt.json' in files:
            with open(os.path.join(root, 'prompt.json'), 'r') as f:
                prompt_data = json.load(f)
                # DragBench prompt.json format: {"source": [[y, x]], "target": [[y, x]]}
                # We need [x, y] for DragGAN
                handle_points = [[p[1], p[0]] for p in prompt_data.get('source', [])]
                target_points = [[p[1], p[0]] for p in prompt_data.get('target', [])]
            
            # Try to load mask.png if it exists
            if 'mask.png' in files:
                mask_img = Image.open(os.path.join(root, 'mask.png')).convert('L')
                mask_data = np.array(mask_img)
                # Convert to 0/1 (DragBench masks are usually 255 for fixed, 0 for moving)
                # In DragGAN, mask=1 means fixed (preserved), mask=0 means flexible
                mask_data = (mask_data > 128).astype(np.uint8)
        
        if handle_points and target_points:
            cases.append({
                'points': handle_points,
                'targets': target_points,
                'mask': mask_data,
                'pkl': assigned_pkl,
                'name': os.path.basename(root),
                'category': matched_kw,
                'source_res': source_res
            })
    
    # Summary of loaded cases
    if cases:
        counts = {}
        for c in cases:
            counts[c['category']] = counts.get(c['category'], 0) + 1
        print(f"    Loaded {len(cases)} cases across categories: {counts}")
    
    return cases

def generate_mock_cases(num=10):
    # If DragBench is missing, we return empty list
    print("(!) DragBench data is missing and mock generation is disabled.")
    return []

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='datasets/dragbench', help='Path to DragBench data')
    # Change --steps to accept multiple integers
    parser.add_argument('--steps_list', type=int, nargs='+', default=[30, 50, 80], help='List of steps to evaluate')
    parser.add_argument('--num', type=int, default=10, help='Num cases if data is missing')
    parser.add_argument('--output', type=str, default='eval_results_detailed.json', help='Output JSON file')
    args_cmd = parser.parse_args()

    # Set random seed
    set_random_seed(42)

    print(f"\n[Evaluation] Target: DragBench | Steps: {args_cmd.steps_list}")
    
    test_cases = load_dragbench(args_cmd.data)
    if not test_cases:
        print(f"(!) DragBench data not found at '{args_cmd.data}'. Using {args_cmd.num} random cases for comparison.")
        test_cases = generate_mock_cases(args_cmd.num)
    else:
        print(f"(+) Found {len(test_cases)} DragBench cases.")
        print(f"Sample case: {test_cases[0]}")

    eval_configs = [
        {"tracker": "Baseline", "mask": "Baseline", "label": "Baseline (Nearest Neighbor)", "use_mask": False},
        {"tracker": "WCAT", "mask": "Baseline", "label": "WCAT (Weighted Context-Aware)", "use_mask": False},
    ]
    
    # Structure to hold all results: {step: {method: [results]}}
    comprehensive_results = {}
    
    # Base directory for images
    eval_base_dir = 'eval_results_dragbench'
    # Clean up only if we are starting a fresh full run (optional, but safe to keep clean)
    if os.path.exists(eval_base_dir):
        shutil.rmtree(eval_base_dir)
    os.makedirs(eval_base_dir)
    
    # Iterate over each step count
    for steps in args_cmd.steps_list:
        print(f"\n{'='*40} Evaluating at {steps} Steps {'='*40}")
        step_results = {}
        step_fid = {}
        
        print("-" * 120)
        print(f"{'Method':<35} | {'Mean MD':<15} | {'FID':<15}")
        print("-" * 120)
        
        for cfg in eval_configs:
            # Create a specific directory for this method and step count
            # e.g. eval_results_dragbench/steps_30/Baseline
            method_save_dir = os.path.join(eval_base_dir, f"steps_{steps}", cfg['label'].replace(' ', '_'))
            
            results = run_eval(cfg['tracker'], test_cases, steps=steps, 
                             mask_handler_name=cfg['mask'], use_mask=cfg.get('use_mask', True),
                             save_dir=method_save_dir)
            
            avg_md = np.mean([r['md'] for r in results]) if results else 0
            step_results[cfg['label']] = results
            
            # Calculate FID
            # Note: Reference images are shared, but we save them per step-run to be safe or use a common ref dir
            # Here we used 'reference' inside the parent of method_save_dir which is steps_{steps}/reference
            ref_dir = os.path.join(eval_base_dir, f"steps_{steps}", 'reference')
            ref_paths = [os.path.join(ref_dir, f) for f in os.listdir(ref_dir) if f.endswith('.png')] if os.path.exists(ref_dir) else []
            eval_paths = [os.path.join(method_save_dir, f) for f in os.listdir(method_save_dir) if f.endswith('.png')] if os.path.exists(method_save_dir) else []
            
            fid_val = float('nan')
            if len(ref_paths) >= 2 and len(eval_paths) >= 2:
                try:
                    fid_val = compute_fid_from_paths(ref_paths, eval_paths, batch_size=min(len(eval_paths), 32))
                except Exception as e:
                    print(f"  Error calculating FID for {cfg['label']}: {e}")
            else:
                if len(eval_paths) > 0:
                    print(f"  Skipping FID for {cfg['label']} (not enough samples: ref={len(ref_paths)}, eval={len(eval_paths)})")
            
            step_fid[cfg['label']] = fid_val
            print(f"{cfg['label']:<35} | {avg_md:<15.4f} | {fid_val:<15.4f}")
        
        print("-" * 120)
        
        comprehensive_results[steps] = {
            'detailed': step_results,
            'fid': step_fid
        }

    # Print Detailed Breakdown for the last run (or all, but that's too long)
    # We will save everything to JSON
    
    # Save results to JSON
    with open(args_cmd.output, 'w') as f:
        json.dump(comprehensive_results, f, indent=4)
    print(f"\n(+) Detailed results and FID for all steps saved to {args_cmd.output}")
