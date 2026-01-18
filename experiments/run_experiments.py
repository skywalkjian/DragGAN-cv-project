import os
import sys
import torch
import numpy as np
import json
import random
import argparse
import warnings
import shutil
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import dnnlib

from viz.renderer import Renderer
from viz.trackers import get_tracker
from viz.mask_handlers import get_mask_handler
from calculate_fid import compute_fid_from_paths

warnings.filterwarnings("ignore")

def set_random_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_comparison_collage(case_name, images, save_path):
    """
    images: list of (title, PIL.Image)
    """
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    
    if n == 1:
        axes = [axes]
        
    for i, (title, img) in enumerate(images):
        axes[i].imshow(img)
        axes[i].set_title(title, fontsize=15)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"(+) Comparison collage saved to {save_path}")

def load_dataset(path, filter_list=None):
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
        'turtle': 'checkpoints/stylegan2-ffhq-512x512.pkl',
    }

    cases = []
    if not os.path.exists(path):
        return []
        
    for root, dirs, files in os.walk(path):
        case_name = os.path.basename(root)
        if filter_list and case_name not in filter_list:
            continue
            
        handle_points = []
        target_points = []
        mask_data = None
        source_res = None
        
        path_lower = root.lower()
        assigned_pkl = None
        matched_kw = None
        
        for kw, pkl_path in CATEGORY_TO_PKL.items():
            if kw in path_lower:
                if os.path.exists(pkl_path):
                    assigned_pkl = pkl_path
                    matched_kw = kw
                    break
        
        if not assigned_pkl and ('generated' in path_lower or 'nature' in path_lower or 'art' in path_lower):
             if os.path.exists('checkpoints/stylegan2-ffhq-512x512.pkl'):
                 assigned_pkl = 'checkpoints/stylegan2-ffhq-512x512.pkl'
                 matched_kw = 'generic'

        if not assigned_pkl:
            continue

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
                handle_points = [[p[1], p[0]] for p in prompt_data.get('source', [])]
                target_points = [[p[1], p[0]] for p in prompt_data.get('target', [])]
            
            if 'mask.png' in files:
                mask_img = Image.open(os.path.join(root, 'mask.png')).convert('L')
                mask_data = np.array(mask_img)
                mask_data = (mask_data > 128).astype(np.uint8)
        
        if handle_points and target_points:
            cases.append({
                'points': handle_points,
                'targets': target_points,
                'mask': mask_data,
                'pkl': assigned_pkl,
                'name': case_name,
                'category': matched_kw,
                'source_res': source_res
            })
    
    return cases

def run_eval(renderer, config_name, test_cases, steps=50, mask_handler_name='Baseline', use_mask=True, save_dir=None):
    results = []
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        ref_dir = os.path.join(os.path.dirname(save_dir), 'reference')
        os.makedirs(ref_dir, exist_ok=True)

    for idx, case in enumerate(tqdm(test_cases, desc=f"Eval {config_name}")):
        case_name = case['name']
        pkl_path = case['pkl']
        points = case['points']
        targets = case['targets']
        
        if 'source_res' in case and case['source_res']:
            target_res = renderer.get_resolution(pkl_path)
            src_h, src_w = case['source_res']
            scale_h, scale_w = target_res / src_h, target_res / src_w
            points = [[p[0] * scale_h, p[1] * scale_w] for p in points]
            targets = [[p[0] * scale_h, p[1] * scale_w] for p in targets]
        
        seed = case.get('seed', 0)
        args = dnnlib.EasyDict(
            pkl=pkl_path, w0_seed=seed, batch_size=1, lr=0.001, w_plus=True,
            reset_w=True, is_drag=True, points=points, targets=targets,
            mask=case.get('mask', None) if use_mask else None,
            feature_idx=5, tracker_type=config_name
        )
        
        renderer.tracker = get_tracker(config_name)
        renderer.mask_handler = get_mask_handler(mask_handler_name)
        
        res0 = renderer.render(**args)
        if save_dir:
            ref_path = os.path.join(os.path.dirname(save_dir), 'reference', f"{case_name}_orig.png")
            if not os.path.exists(ref_path):
                img0 = res0.image
                img0 = Image.fromarray(np.array(img0).astype(np.uint8))
                if img0.width > 512 or img0.height > 512:
                    img0.thumbnail((512, 512), Image.LANCZOS)
                img0.save(ref_path)
        
        current_points = [p[:] for p in points]
        args.reset_w = False
        last_md = 0
        
        for i in range(steps):
            args.points = current_points
            res = renderer.render(**args)
            if 'error' in res and res.error: break
            if 'md' in res: last_md = res.md
            if 'points' in res: current_points = [[p[0], p[1]] for p in res.points]
            
            if i == steps - 1:
                if save_dir and 'image' in res:
                    img = Image.fromarray(np.array(res.image).astype(np.uint8))
                    if img.width > 512 or img.height > 512:
                        img.thumbnail((512, 512), Image.LANCZOS)
                    img.save(os.path.join(save_dir, f"{case_name}_final.png"))

        results.append({'name': case_name, 'md': float(last_md)})
    return results

def run_eval_stability(renderer, config_name, test_cases, steps=50, mask_handler_name='Baseline', use_mask=True, save_dir=None):
    results = []
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        diff_dir = os.path.join(save_dir, 'diff_maps')
        os.makedirs(diff_dir, exist_ok=True)

    for idx, case in enumerate(tqdm(test_cases, desc=f"Stability {mask_handler_name}")):
        case_name = case['name']
        pkl_path = case['pkl']
        points = case['points']
        targets = case['targets']
        
        if 'source_res' in case and case['source_res']:
            target_res = renderer.get_resolution(pkl_path)
            src_h, src_w = case['source_res']
            scale_h, scale_w = target_res / src_h, target_res / src_w
            points = [[p[0] * scale_h, p[1] * scale_w] for p in points]
            targets = [[p[0] * scale_h, p[1] * scale_w] for p in targets]
        
        seed = case.get('seed', 0)
        args = dnnlib.EasyDict(
            pkl=pkl_path, w0_seed=seed, batch_size=1, lr=0.001, w_plus=True,
            reset_w=True, is_drag=True, points=points, targets=targets,
            mask=case.get('mask', None) if use_mask else None,
            feature_idx=5, tracker_type=config_name
        )
        
        renderer.tracker = get_tracker(config_name)
        renderer.mask_handler = get_mask_handler(mask_handler_name)
        
        res0 = renderer.render(**args)
        img0 = np.array(res0.image).astype(np.float32)
        current_points = [p[:] for p in points]
        args.reset_w = False
        prev_img = img0
        stability_score = 0.0
        
        for i in range(steps):
            args.points = current_points
            res = renderer.render(**args)
            if 'points' in res: current_points = [[p[0], p[1]] for p in res.points]
            curr_img = np.array(res.image).astype(np.float32)
            step_diff = np.mean(np.abs(curr_img - prev_img))
            stability_score += step_diff
            prev_img = curr_img

        stability_score /= steps
        final_img_pil = Image.fromarray(curr_img.astype(np.uint8))
        diff_map = np.abs(curr_img - img0).mean(axis=2)
        
        # Normalize diff map for visualization
        diff_map_norm = (diff_map - diff_map.min()) / (diff_map.max() - diff_map.min() + 1e-8)
        diff_map_vis = (plt.cm.hot(diff_map_norm)[:, :, :3] * 255).astype(np.uint8)
        diff_map_pil = Image.fromarray(diff_map_vis)

        # Create mask overlay
        orig_img_np = img0.astype(np.uint8)
        if orig_img_np.shape[2] == 4:
            orig_img_np = orig_img_np[:, :, :3]
            
        mask_overlay = orig_img_np.copy()
        if case.get('mask', None) is not None:
            mask = case['mask']
            # Resize mask if needed
            if mask.shape != orig_img_np.shape[:2]:
                mask_pil = Image.fromarray(mask).resize((orig_img_np.shape[1], orig_img_np.shape[0]), Image.NEAREST)
                mask = np.array(mask_pil)
            
            # Blend with red
            mask_bool = mask > 0
            mask_overlay[mask_bool] = (mask_overlay[mask_bool].astype(np.float32) * 0.5 + 
                                       np.array([255, 0, 0], dtype=np.float32) * 0.5).astype(np.uint8)
        mask_overlay_pil = Image.fromarray(mask_overlay)

        if save_dir:
            final_img_pil.save(os.path.join(save_dir, f"{case_name}_final.png"))
            plt.figure(figsize=(5, 5)); plt.imshow(diff_map, cmap='hot'); plt.axis('off')
            plt.savefig(os.path.join(diff_dir, f"{case_name}_diff.png"), bbox_inches='tight', pad_inches=0)
            plt.close()

        results.append({
            'name': case_name, 
            'stability': float(stability_score),
            'final_img': final_img_pil,
            'diff_map': diff_map_pil,
            'orig_img': Image.fromarray(img0.astype(np.uint8)),
            'mask_overlay': mask_overlay_pil
        })
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=int, default=1, choices=[1, 2, 3, 4], help='Experiment ID')
    parser.add_argument('--steps', type=int, default=50, help='Optimization steps')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    set_random_seed(args.seed)
    renderer = Renderer(disable_timing=True)
    
    # Dataset configurations
    DRAGBENCH_PATH = 'experiments/data/dragbench'
    CUSTOM_PATH = 'experiments/data/custom'
    
    if args.exp == 1:
        # Exp 1: DragBench Tracker Evaluation
        print("\n--- Running Experiment 1: DragBench Tracker Evaluation ---")
        test_cases = load_dataset(DRAGBENCH_PATH)
        output_dir = 'experiments/results/exp1_dragbench_tracker'
        eval_configs = [
            {"tracker": "Baseline", "mask": "Baseline", "label": "Baseline", "use_mask": False},
            {"tracker": "WCAT", "mask": "Baseline", "label": "WCAT", "use_mask": False},
        ]
    elif args.exp == 2:
        # Exp 2: Custom Dataset Tracker Evaluation
        print("\n--- Running Experiment 2: Custom Dataset Tracker Evaluation ---")
        test_cases = load_dataset(CUSTOM_PATH)
        output_dir = 'experiments/results/exp2_custom_tracker'
        eval_configs = [
            {"tracker": "Baseline", "mask": "Baseline", "label": "Baseline", "use_mask": False},
            {"tracker": "WCAT", "mask": "Baseline", "label": "WCAT", "use_mask": False},
        ]
    elif args.exp == 3:
        # Exp 3: Mask Stability Evaluation (5 specific samples)
        print("\n--- Running Experiment 3: Mask Stability Evaluation ---")
        filter_list = ['cat_0', 'dog_3', 'elephant_0', 'horse_2', 'lion_2']
        # Also include available ones if some are missing
        test_cases = load_dataset(DRAGBENCH_PATH, filter_list=filter_list)
        if not test_cases:
            print("Warning: Specific samples not found, using first 5 from DragBench")
            test_cases = load_dataset(DRAGBENCH_PATH)[:5]
        output_dir = 'experiments/results/exp3_mask_stability'
        eval_configs = [
            {"tracker": "Baseline", "mask": "Baseline", "label": "Baseline", "use_mask": True},
            {"tracker": "Baseline", "mask": "Loss Scheduling", "label": "Loss_Scheduling", "use_mask": True},
        ]
    elif args.exp == 4:
        # Exp 4: DragBench Mask Evaluation
        print("\n--- Running Experiment 4: DragBench Mask Evaluation ---")
        test_cases = load_dataset(DRAGBENCH_PATH)
        output_dir = 'experiments/results/exp4_dragbench_mask'
        eval_configs = [
            {"tracker": "Baseline", "mask": "Baseline", "label": "Baseline", "use_mask": True},
            {"tracker": "Baseline", "mask": "Loss Scheduling", "label": "Loss_Scheduling", "use_mask": True},
        ]

    if not test_cases:
        print(f"Error: No test cases found.")
        exit(1)
    
    print(f"Loaded {len(test_cases)} cases.")
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    all_results = {}
    fid_scores = {}
    
    print("-" * 80)
    if args.exp == 3:
        print(f"{'Method':<20} | {'Mean Stability':<20}")
    else:
        print(f"{'Method':<20} | {'Mean MD':<15} | {'FID':<15}")
    print("-" * 80)

    for cfg in eval_configs:
        method_save_dir = os.path.join(output_dir, cfg['label'])
        
        if args.exp == 3:
            results = run_eval_stability(renderer, cfg['tracker'], test_cases, steps=args.steps, 
                                          mask_handler_name=cfg['mask'], use_mask=cfg['use_mask'],
                                          save_dir=method_save_dir)
            avg_stab = np.mean([r['stability'] for r in results])
            all_results[cfg['label']] = results
            print(f"{cfg['label']:<20} | {avg_stab:<20.4f}")
        else:
            results = run_eval(renderer, cfg['tracker'], test_cases, steps=args.steps, 
                              mask_handler_name=cfg['mask'], use_mask=cfg['use_mask'],
                              save_dir=method_save_dir)
            avg_md = np.mean([r['md'] for r in results])
            all_results[cfg['label']] = results
            
            # FID calculation
            ref_dir = os.path.join(output_dir, 'reference')
            ref_paths = [os.path.join(ref_dir, f) for f in os.listdir(ref_dir) if f.endswith('.png')]
            eval_paths = [os.path.join(method_save_dir, f) for f in os.listdir(method_save_dir) if f.endswith('.png')]
            
            fid_val = float('nan')
            if len(ref_paths) >= 2 and len(eval_paths) >= 2:
                try:
                    fid_val = compute_fid_from_paths(ref_paths, eval_paths, batch_size=min(len(eval_paths), 32))
                except: pass
            
            fid_scores[cfg['label']] = fid_val
            print(f"{cfg['label']:<20} | {avg_md:<15.4f} | {fid_val:<15.4f}")

    # Save results
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        # Convert PIL images to paths or remove them for JSON serialization
        serializable_results = {}
        for k, v in all_results.items():
            serializable_results[k] = [{kk: vv for kk, vv in r.items() if not isinstance(vv, Image.Image)} for r in v]
        json.dump({'detailed': serializable_results, 'fid': fid_scores if args.exp != 3 else None}, f, indent=4)

    # Generate collages for Exp 3
    if args.exp == 3:
        print("\n--- Generating Comparison Collages for Exp 3 ---")
        collage_dir = os.path.join(output_dir, 'collages')
        os.makedirs(collage_dir, exist_ok=True)
        
        methods = list(all_results.keys())
        for i in range(len(test_cases)):
            case_name = test_cases[i]['name']
            
            # Prepare images for collage
            # Format: [Original, Mask Overlay, Method1_Final, Method2_Final, Method1_Diff, Method2_Diff]
            collage_images = []

            # Original Image
            orig_img = all_results[methods[0]][i]['orig_img']
            collage_images.append(("Original", orig_img))
            
            # Mask Overlay (from the first method's result for this case)
            mask_overlay = all_results[methods[0]][i]['mask_overlay']
            collage_images.append(("Mask Visualization", mask_overlay))
            
            # Final images for each method
            for m in methods:
                final_img = all_results[m][i]['final_img']
                collage_images.append((f"{m} Final", final_img))
            
            # Diff maps for each method
            for m in methods:
                diff_map = all_results[m][i]['diff_map']
                collage_images.append((f"{m} Diff", diff_map))
                
            save_path = os.path.join(collage_dir, f"{case_name}_collage.png")
            save_comparison_collage(case_name, collage_images, save_path)
