import os
import json
import torch
import numpy as np
from PIL import Image
import dnnlib
from viz.renderer import Renderer
from gradio_utils.utils import draw_points_on_image

def create_custom_dragbench():
    # Define the 16 cases from user input
    # Note: Gradio uses [x, y], DragGAN internal uses [y, x]
    raw_cases = [
        {'name': 'lion_g1', 'pkl': 'checkpoints/stylegan2_lions_512_pytorch.pkl', 'seed': 0, 'pts': [18, 446], 'tgs': [194, 506]},
        {'name': 'lion_g2', 'pkl': 'checkpoints/stylegan2_lions_512_pytorch.pkl', 'seed': 0, 'pts': [354, 268], 'tgs': [303, 343]},
        {'name': 'afhqcat_g1', 'pkl': 'checkpoints/stylegan2-afhqcat-512x512.pkl', 'seed': 0, 'pts': [318, 351], 'tgs': [387, 353]},
        {'name': 'afhqcat_g2', 'pkl': 'checkpoints/stylegan2-afhqcat-512x512.pkl', 'seed': 0, 'pts': [254, 312], 'tgs': [255, 259]},
        {'name': 'car_g1', 'pkl': 'checkpoints/stylegan2-car-config-f.pkl', 'seed': 0, 'pts': [136, 253], 'tgs': [174, 262]},
        {'name': 'car_g2', 'pkl': 'checkpoints/stylegan2-car-config-f.pkl', 'seed': 0, 'pts': [311, 241], 'tgs': [347, 254]},
        {'name': 'cat_g1', 'pkl': 'checkpoints/stylegan2-cat-config-f.pkl', 'seed': 0, 'pts': [152, 176], 'tgs': [139, 190]},
        {'name': 'cat_g2', 'pkl': 'checkpoints/stylegan2-cat-config-f.pkl', 'seed': 0, 'pts': [207, 166], 'tgs': [212, 135]},
        {'name': 'ffhq_g1', 'pkl': 'checkpoints/stylegan2-ffhq-512x512.pkl', 'seed': 0, 'pts': [259, 80], 'tgs': [250, 108]},
        {'name': 'ffhq_g2', 'pkl': 'checkpoints/stylegan2-ffhq-512x512.pkl', 'seed': 0, 'pts': [249, 205], 'tgs': [251, 165]},
        {'name': 'dog_g1', 'pkl': 'checkpoints/stylegan2_dogs_1024_pytorch.pkl', 'seed': 0, 'pts': [856, 284], 'tgs': [762, 456]},
        {'name': 'dog_g2', 'pkl': 'checkpoints/stylegan2_dogs_1024_pytorch.pkl', 'seed': 0, 'pts': [252, 645], 'tgs': [310, 502]},
        {'name': 'elephants_g1', 'pkl': 'checkpoints/stylegan2_elephants_512_pytorch.pkl', 'seed': 0, 'pts': [134, 247], 'tgs': [210, 243]},
        {'name': 'elephants_g2', 'pkl': 'checkpoints/stylegan2_elephants_512_pytorch.pkl', 'seed': 0, 'pts': [300, 332], 'tgs': [300, 259]},
        {'name': 'horse_g1', 'pkl': 'checkpoints/stylegan2_horses_256_pytorch.pkl', 'seed': 0, 'pts': [241, 147], 'tgs': [244, 105]},
        {'name': 'horse_g2', 'pkl': 'checkpoints/stylegan2_horses_256_pytorch.pkl', 'seed': 0, 'pts': [165, 114], 'tgs': [125, 101]},
    ]

    base_dir = '/root/DragGAN-cv-project/datasets/custom_dragbench'
    os.makedirs(base_dir, exist_ok=True)
    renderer = Renderer(disable_timing=True)

    print(f"Creating Custom DragBench at {base_dir}...")

    for case in raw_cases:
        case_path = os.path.join(base_dir, case['name'])
        os.makedirs(case_path, exist_ok=True)

        # 1. Save prompt.json (DragBench format)
        # DragBench usually stores [y, x] in prompt.json
        prompt_data = {
            "source": [[case['pts'][1], case['pts'][0]]],
            "target": [[case['tgs'][1], case['tgs'][0]]],
            "prompt": f"A custom case for {case['name']}",
            "seed": case['seed']
        }
        with open(os.path.join(case_path, 'prompt.json'), 'w') as f:
            json.dump(prompt_data, f)

        # 2. Generate and save original image
        args = dnnlib.EasyDict(
            pkl=case['pkl'],
            w0_seed=case['seed'],
            batch_size=1,
            lr=0.001,
            w_plus=True,
            reset_w=True,
            is_drag=False
        )
        res = renderer.render(**args)
        img = Image.fromarray(res.image.astype(np.uint8))
        img.save(os.path.join(case_path, 'origin_image.png'))

        # 3. Generate and save Dot visualization (Red/Blue/Yellow)
        # draw_points_on_image expects [x, y]
        points_dict = {
            0: {
                'start': [case['pts'][0], case['pts'][1]],
                'target': [case['tgs'][0], case['tgs'][1]]
            }
        }
        viz_img = draw_points_on_image(img, points_dict)
        viz_img.save(os.path.join(case_path, 'visualized_dots.png'))
        
        print(f"  Processed {case['name']}")

if __name__ == "__main__":
    create_custom_dragbench()
