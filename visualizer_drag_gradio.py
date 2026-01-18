import os
import os.path as osp
from argparse import ArgumentParser
from functools import partial

import gradio as gr
import numpy as np
import torch
from PIL import Image

import dnnlib
from gradio_utils import (ImageMask, draw_mask_on_image, draw_points_on_image,
                          get_latest_points_pair, get_valid_mask,
                          on_change_single_global_state)
from viz.renderer import Renderer, add_watermark_np

parser = ArgumentParser()
parser.add_argument('--share', action='store_true',default='True')
parser.add_argument('--cache-dir', type=str, default='./checkpoints')
parser.add_argument(
    "--listen",
    action="store_true",
    help="launch gradio with 0.0.0.0 as server name, allowing to respond to network requests",
)
args = parser.parse_args()

cache_dir = args.cache_dir

device = 'cuda'


def reverse_point_pairs(points):
    new_points = []
    for p in points:
        new_points.append([p[1], p[0]])
    return new_points


def clear_state(global_state, target=None):
    """Clear target history state from global_state
    If target is not defined, points and mask will be both removed.
    1. set global_state['points'] as empty dict
    2. set global_state['mask'] as full-one mask.
    """
    if target is None:
        target = ['point', 'mask']
    if not isinstance(target, list):
        target = [target]
    if 'point' in target:
        global_state['points'] = dict()
        print('Clear Points State!')
    if 'mask' in target:
        image_raw = global_state["images"]["image_raw"]
        global_state['mask'] = np.ones((image_raw.size[1], image_raw.size[0]),
                                       dtype=np.uint8)
        print('Clear mask State!')

    return global_state


def init_images(global_state):
    """This function is called only ones with Gradio App is started.
    0. pre-process global_state, unpack value from global_state of need
    1. Re-init renderer
    2. run `renderer._render_drag_impl` with `is_drag=False` to generate
       new image
    3. Assign images to global state and re-generate mask
    """

    if isinstance(global_state, gr.State):
        state = global_state.value
    else:
        state = global_state

    if not isinstance(state['generator_params'], dnnlib.EasyDict):
        state['generator_params'] = dnnlib.EasyDict(state['generator_params'])

    state['renderer'].init_network(
        state['generator_params'],  # res
        valid_checkpoints_dict[state['pretrained_weight']],  # pkl
        state['params']['seed'],  # w0_seed,
        None,  # w_load
        state['params']['latent_space'] == 'w+',  # w_plus
        'const',
        state['params']['trunc_psi'],  # trunc_psi,
        state['params']['trunc_cutoff'],  # trunc_cutoff,
        None,  # input_transform
        state['params']['lr']  # lr,
    )

    state['renderer']._render_drag_impl(state['generator_params'],
                                        is_drag=False,
                                        to_pil=True)

    init_image = state['generator_params'].image
    state['images']['image_orig'] = init_image
    state['images']['image_raw'] = init_image
    state['images']['image_show'] = Image.fromarray(
        add_watermark_np(np.array(init_image)))
    state['mask'] = np.ones((init_image.size[1], init_image.size[0]),
                            dtype=np.uint8)
    return global_state


def update_image_draw(image, points, mask, show_mask, global_state=None):

    image_draw = draw_points_on_image(image, points)
    if show_mask and mask is not None and not (mask == 0).all() and not (
            mask == 1).all():
        image_draw = draw_mask_on_image(image_draw, mask)

    image_draw = Image.fromarray(add_watermark_np(np.array(image_draw)))
    if global_state is not None:
        global_state['images']['image_show'] = image_draw
    return image_draw


def preprocess_mask_info(global_state, image):
    """Function to handle mask information.
    1. last_mask is None: Do not need to change mask, return mask
    2. last_mask is not None:
        2.1 global_state is remove_mask:
        2.2 global_state is add_mask:
    """
    if isinstance(image, dict):
        last_mask = get_valid_mask(image['mask'])
    else:
        last_mask = None
    mask = global_state['mask']

    # mask in global state is a placeholder with all 1.
    if (mask == 1).all():
        mask = last_mask

    # last_mask = global_state['last_mask']
    editing_mode = global_state['editing_state']

    if last_mask is None:
        return global_state

    if editing_mode == 'remove_mask':
        updated_mask = np.clip(mask - last_mask, 0, 1)
        print(f'Last editing_state is {editing_mode}, do remove.')
    elif editing_mode == 'add_mask':
        updated_mask = np.clip(mask + last_mask, 0, 1)
        print(f'Last editing_state is {editing_mode}, do add.')
    else:
        updated_mask = mask
        print(f'Last editing_state is {editing_mode}, '
              'do nothing to mask.')

    global_state['mask'] = updated_mask
    # global_state['last_mask'] = None  # clear buffer
    return global_state


valid_checkpoints_dict = {
    f.split('/')[-1].split('.')[0]: osp.join(cache_dir, f)
    for f in os.listdir(cache_dir)
    if (f.endswith('pkl') and osp.exists(osp.join(cache_dir, f)))
}
print(f'File under cache_dir ({cache_dir}):')
print(os.listdir(cache_dir))
print('Valid checkpoint file:')
print(valid_checkpoints_dict)

init_pkl = 'stylegan2_lions_512_pytorch'

with gr.Blocks() as app:

    # renderer = Renderer()
    global_state = gr.State({
        "images": {
            # image_orig: the original image, change with seed/model is changed
            # image_raw: image with mask and points, change durning optimization
            # image_show: image showed on screen
        },
        "temporal_params": {
            # stop
        },
        'mask':
        None,  # mask for visualization, 1 for editing and 0 for unchange
        'last_mask': None,  # last edited mask
        'show_mask': True,  # add button
        "generator_params": dnnlib.EasyDict(),
        "params": {
            "seed": 0,
            "motion_lambda": 20,
            "r1_in_pixels": 3,
            "r2_in_pixels": 12,
            "magnitude_direction_in_pixels": 1.0,
            "latent_space": "w+",
            "trunc_psi": 0.7,
            "trunc_cutoff": None,
            "lr": 0.001,
        },
        "device": device,
        "draw_interval": 1,
        "renderer": Renderer(disable_timing=True),
        "points": {},
        "curr_point": None,
        "curr_type_point": "start",
        'editing_state': 'add_points',
        'pretrained_weight': init_pkl,
        'tracker_name': 'DragGAN (Baseline)',
        'mask_handler_name': 'DragGAN (Baseline)'
    })

    # init image
    global_state = init_images(global_state)

    with gr.Row():

        with gr.Row():

            # Left --> tools
            with gr.Column(scale=3):

                # Pickle
                with gr.Row():

                    with gr.Column(scale=1, min_width=10):
                        gr.Markdown(value='Pickle', show_label=False)

                    with gr.Column(scale=4, min_width=10):
                        form_pretrained_dropdown = gr.Dropdown(
                            choices=list(valid_checkpoints_dict.keys()),
                            label="Pretrained Model",
                            value=init_pkl,
                        )

                # Latent
                with gr.Row():
                    with gr.Column(scale=1, min_width=10):
                        gr.Markdown(value='Latent', show_label=False)

                    with gr.Column(scale=4, min_width=10):
                        form_seed_number = gr.Number(
                            value=global_state.value['params']['seed'],
                            interactive=True,
                            label="Seed",
                        )
                        form_lr_number = gr.Number(
                            value=global_state.value["params"]["lr"],
                            interactive=True,
                            label="Step Size")

                        with gr.Row():
                            with gr.Column(scale=2, min_width=10):
                                form_reset_image = gr.Button("Reset Image")
                            with gr.Column(scale=3, min_width=10):
                                form_latent_space = gr.Radio(
                                    ['w', 'w+'],
                                    value=global_state.value['params']
                                    ['latent_space'],
                                    interactive=True,
                                    label='Latent space to optimize',
                                    show_label=False,
                                )

                # Drag
                with gr.Row():
                    with gr.Column(scale=1, min_width=10):
                        gr.Markdown(value='Drag', show_label=False)
                    with gr.Column(scale=4, min_width=10):
                        form_tracker_dropdown = gr.Dropdown(
                            choices=['DragGAN (Baseline)', 'RAFT Large (Optical Flow)', 'Deep Particle (PIPs-like)', 'WCAT (Weighted Context-Aware Tracker)'],
                            label="Tracking Method",
                            value='DragGAN (Baseline)',
                        )
                        with gr.Row():
                            with gr.Column(scale=1, min_width=10):
                                enable_add_points = gr.Button('Add Points')
                            with gr.Column(scale=1, min_width=10):
                                undo_points = gr.Button('Reset Points')
                        with gr.Row():
                            with gr.Column(scale=1, min_width=10):
                                form_start_btn = gr.Button("Start")
                            with gr.Column(scale=1, min_width=10):
                                form_stop_btn = gr.Button("Stop")

                        form_steps_number = gr.Number(value=0,
                                                      label="Steps",
                                                      interactive=False)

                # Mask
                with gr.Row():
                    with gr.Column(scale=1, min_width=10):
                        gr.Markdown(value='Mask', show_label=False)
                    with gr.Column(scale=4, min_width=10):
                        form_mask_handler_dropdown = gr.Dropdown(
                            choices=['DragGAN (Baseline)'],
                            label="Masking Method",
                            value='DragGAN (Baseline)',
                        )
                        enable_add_mask = gr.Button('Edit Flexible Area')
                        with gr.Row():
                            with gr.Column(scale=1, min_width=10):
                                form_reset_mask_btn = gr.Button("Reset mask")
                            with gr.Column(scale=1, min_width=10):
                                show_mask = gr.Checkbox(
                                    label='Show Mask',
                                    value=global_state.value['show_mask'],
                                    show_label=False)

                        with gr.Row():
                            form_lambda_number = gr.Number(
                                value=global_state.value["params"]
                                ["motion_lambda"],
                                interactive=True,
                                label="Lambda",
                            )

                form_draw_interval_number = gr.Number(
                    value=global_state.value["draw_interval"],
                    label="Draw Interval (steps)",
                    interactive=True,
                    visible=False)

            # Right --> Image
            with gr.Column(scale=8):
                form_image = ImageMask(
                    value=global_state.value['images']['image_show'],
                    brush_radius=20,
                    elem_id="image_mask_output")
                
                with gr.Accordion("Difference Map (Background Consistency)", open=False):
                    form_diff_image = gr.Image(
                        label="Difference from initial image (5x boosted)",
                        interactive=False,
                        visible=True,
                        elem_id="diff_image_output"
                    )
    gr.Markdown("""
        ## Quick Start

        1. Select desired `Pretrained Model` and adjust `Seed` to generate an
           initial image.
        2. Click on image to add control points.
        3. Click `Start` and enjoy it!

        ## Advance Usage

        1. Change `Step Size` to adjust learning rate in drag optimization.
        2. Select `w` or `w+` to change latent space to optimize:
        * Optimize on `w` space may cause greater influence to the image.
        * Optimize on `w+` space may work slower than `w`, but usually achieve
          better results.
        * Note that changing the latent space will reset the image, points and
          mask (this has the same effect as `Reset Image` button).
        3. Click `Edit Flexible Area` to create a mask and constrain the
           unmasked region to remain unchanged.
        """)
    gr.HTML("""
        <style>
            .container {
                position: absolute;
                height: 50px;
                text-align: center;
                line-height: 50px;
                width: 100%;
            }
            #image_mask_output, #diff_image_output {
                max-width: 512px !important;
            }
            /* Ensure the canvas/image doesn't exceed 512x512 but allows space for toolbars */
            #image_mask_output img, #image_mask_output canvas {
                max-width: 512px !important;
                max-height: 512px !important;
                object-fit: contain;
            }
        </style>
        <div class="container">
        Gradio demo supported by
        <img src="https://avatars.githubusercontent.com/u/10245193?s=200&v=4" height="20" width="20" style="display:inline;">
        <a href="https://github.com/open-mmlab/mmagic">OpenMMLab MMagic</a>
        </div>
        """)

    # Network & latents tab listeners
    def on_change_pretrained_dropdown(pretrained_value, global_state):
        """Function to handle model change.
        1. Set pretrained value to global_state
        2. Re-init images and clear all states
        """

        global_state['pretrained_weight'] = pretrained_value
        init_images(global_state)
        clear_state(global_state)

        return global_state, global_state["images"]['image_show']

    form_pretrained_dropdown.change(
        on_change_pretrained_dropdown,
        inputs=[form_pretrained_dropdown, global_state],
        outputs=[global_state, form_image],
    )

    def on_click_reset_image(global_state):
        """Reset image to the original one and clear all states
        1. Re-init images
        2. Clear all states
        """

        init_images(global_state)
        clear_state(global_state)

        return global_state, global_state['images']['image_show']

    form_reset_image.click(
        on_click_reset_image,
        inputs=[global_state],
        outputs=[global_state, form_image],
    )

    # Update parameters
    def on_change_update_image_seed(seed, global_state):
        """Function to handle generation seed change.
        1. Set seed to global_state
        2. Re-init images and clear all states
        """

        global_state["params"]["seed"] = int(seed)
        init_images(global_state)
        clear_state(global_state)

        return global_state, global_state['images']['image_show']

    form_seed_number.change(
        on_change_update_image_seed,
        inputs=[form_seed_number, global_state],
        outputs=[global_state, form_image],
    )

    def on_click_latent_space(latent_space, global_state):
        """Function to reset latent space to optimize.
        NOTE: this function we reset the image and all controls
        1. Set latent-space to global_state
        2. Re-init images and clear all state
        """

        global_state['params']['latent_space'] = latent_space
        init_images(global_state)
        clear_state(global_state)

        return global_state, global_state['images']['image_show']

    form_latent_space.change(on_click_latent_space,
                             inputs=[form_latent_space, global_state],
                             outputs=[global_state, form_image])

    # ==== Params
    form_lambda_number.change(
        partial(on_change_single_global_state, ["params", "motion_lambda"]),
        inputs=[form_lambda_number, global_state],
        outputs=[global_state],
    )

    def on_change_lr(lr, global_state):
        if lr == 0:
            print('lr is 0, do nothing.')
            return global_state
        else:
            global_state["params"]["lr"] = lr
            renderer = global_state['renderer']
            renderer.update_lr(lr)
            print('New optimizer: ')
            print(renderer.w_optim)
        return global_state

    form_lr_number.change(
        on_change_lr,
        inputs=[form_lr_number, global_state],
        outputs=[global_state],
    )

    def on_change_tracker(tracker_name, global_state):
        global_state['tracker_name'] = tracker_name
        # Update tracker in renderer for real-time switching
        global_state['renderer'].set_tracker(tracker_name)
        return global_state

    form_tracker_dropdown.change(
        on_change_tracker,
        inputs=[form_tracker_dropdown, global_state],
        outputs=[global_state],
    )

    def on_change_mask_handler(mask_handler_name, global_state):
        global_state['mask_handler_name'] = mask_handler_name
        return global_state

    form_mask_handler_dropdown.change(
        on_change_mask_handler,
        inputs=[form_mask_handler_dropdown, global_state],
        outputs=[global_state],
    )

    def on_click_start(global_state, image):
        p_in_pixels = []
        t_in_pixels = []
        valid_points = []

        # handle of start drag in mask editing mode
        global_state = preprocess_mask_info(global_state, image)

        # Prepare the points for the inference
        if len(global_state["points"]) == 0:
            # yield on_click_start_wo_points(global_state, image)
            image_raw = global_state['images']['image_raw']
            update_image_draw(
                image_raw,
                global_state['points'],
                global_state['mask'],
                global_state['show_mask'],
                global_state,
            )

            yield (
                global_state,
                0,
                global_state['images']['image_show'],
                # gr.File.update(visible=False),
                gr.Button.update(interactive=True),
                gr.Button.update(interactive=True),
                gr.Button.update(interactive=True),
                gr.Button.update(interactive=True),
                gr.Button.update(interactive=True),
                # latent space
                gr.Radio.update(interactive=True),
                gr.Button.update(interactive=True),
                # NOTE: disable stop button
                gr.Button.update(interactive=False),

                # update other comps
                gr.Dropdown.update(interactive=True), # Pretrained Model
                gr.Dropdown.update(interactive=True), # Tracking Method
                gr.Dropdown.update(interactive=True), # Masking Method
                gr.Number.update(interactive=True),
                gr.Number.update(interactive=True),
                gr.Button.update(interactive=True),
                gr.Button.update(interactive=True),
                gr.Checkbox.update(interactive=True),
                # gr.Number.update(interactive=True),
                gr.Number.update(interactive=True),
            )
        else:

            # Transform the points into torch tensors
            for key_point, point in global_state["points"].items():
                try:
                    p_start = point.get("start_temp", point["start"])
                    p_end = point["target"]

                    if p_start is None or p_end is None:
                        continue

                    # Gradio image output format: [y, x]
                    p_in_pixels.append([p_start[1], p_start[0]])
                    t_in_pixels.append([p_end[1], p_end[0]])
                    valid_points.append(key_point)
                except KeyError:
                    continue

            # Update tracker name in renderer before starting
            global_state['renderer'].set_tracker(global_state['tracker_name'])

            # Optimization loop
            step = 0
            while True:
                if global_state['temporal_params'].get('stop', False):
                    break
                
                # Perform one step of drag optimization
                # This uses the unified renderer logic with the selected tracker
                res = global_state['renderer']._render_drag_impl(
                    global_state['generator_params'],
                    is_drag=True,
                    points=p_in_pixels,
                    targets=t_in_pixels,
                    mask=global_state['mask'],
                    lambda_mask=global_state['params']['motion_lambda'],
                    reg=0,
                    to_pil=True
                )
                
                # Update points for next iteration based on tracker output
                p_in_pixels = res['points']
                
                # Update global state points for visualization
                for i, key in enumerate(valid_points):
                    global_state['points'][key]['start_temp'] = [p_in_pixels[i][1], p_in_pixels[i][0]]

                step += 1
                if step % global_state['draw_interval'] == 0:
                    image_show = update_image_draw(
                        res['image'],
                        global_state['points'],
                        global_state['mask'],
                        global_state['show_mask'],
                        global_state
                    )
                    
                    # Calculate difference map if needed (optional visualization)
                    # For now, just yield the current image
                    yield (
                        global_state,
                        step,
                        image_show,
                        gr.Button.update(interactive=False),
                        gr.Button.update(interactive=False),
                        gr.Button.update(interactive=False),
                        gr.Button.update(interactive=False),
                        gr.Button.update(interactive=False),
                        gr.Radio.update(interactive=False),
                        gr.Button.update(interactive=False),
                        gr.Button.update(interactive=True),
                        gr.Dropdown.update(interactive=False),
                        gr.Dropdown.update(interactive=False),
                        gr.Dropdown.update(interactive=False),
                        gr.Number.update(interactive=False),
                        gr.Number.update(interactive=False),
                        gr.Button.update(interactive=False),
                        gr.Button.update(interactive=False),
                        gr.Checkbox.update(interactive=False),
                        gr.Number.update(interactive=False),
                    )

            # After stopping, restore interactive UI
            yield (
                global_state,
                step,
                global_state['images']['image_show'],
                gr.Button.update(interactive=True),
                gr.Button.update(interactive=True),
                gr.Button.update(interactive=True),
                gr.Button.update(interactive=True),
                gr.Button.update(interactive=True),
                gr.Radio.update(interactive=True),
                gr.Button.update(interactive=True),
                gr.Button.update(interactive=False),
                gr.Dropdown.update(interactive=True),
                gr.Dropdown.update(interactive=True),
                gr.Dropdown.update(interactive=True),
                gr.Number.update(interactive=True),
                gr.Number.update(interactive=True),
                gr.Button.update(interactive=True),
                gr.Button.update(interactive=True),
                gr.Checkbox.update(interactive=True),
                gr.Number.update(interactive=True),
            )

    form_start_btn.click(
        on_click_start,
        inputs=[global_state, form_image],
        outputs=[
            global_state,
            form_steps_number,
            form_image,
            form_reset_image,
            enable_add_points,
            undo_points,
            form_reset_mask_btn,
            enable_add_mask,
            form_latent_space,
            form_start_btn,
            form_stop_btn,
            form_pretrained_dropdown,
            form_tracker_dropdown,
            form_mask_handler_dropdown,
            form_seed_number,
            form_lr_number,
            form_lambda_number,
            form_draw_interval_number,
            show_mask,
            form_steps_number,
        ],
    )

    def on_click_stop(global_state):
        global_state['temporal_params']['stop'] = True
        return global_state

    form_stop_btn.click(on_click_stop,
                        inputs=[global_state],
                        outputs=[global_state])

    def on_click_remove_point(global_state):
        global_state['points'] = dict()
        global_state['curr_point'] = None
        global_state['curr_type_point'] = 'start'
        return global_state, global_state['images']['image_show']

    undo_points.click(
        on_click_remove_point,
        inputs=[global_state],
        outputs=[global_state, form_image],
    )

    def on_click_reset_mask(global_state):
        global_state['mask'] = np.ones(
            (global_state['images']['image_raw'].size[1],
             global_state['images']['image_raw'].size[0]),
            dtype=np.uint8)
        image_draw = update_image_draw(
            global_state['images']['image_raw'],
            global_state['points'],
            global_state['mask'],
            global_state['show_mask'],
            global_state,
        )
        return global_state, image_show

    form_reset_mask_btn.click(
        on_click_reset_mask,
        inputs=[global_state],
        outputs=[global_state, form_image],
    )

    def on_click_show_mask(show_mask, global_state):
        global_state['show_mask'] = show_mask
        image_draw = update_image_draw(
            global_state['images']['image_raw'],
            global_state['points'],
            global_state['mask'],
            global_state['show_mask'],
            global_state,
        )
        return global_state, image_draw

    show_mask.change(
        on_click_show_mask,
        inputs=[show_mask, global_state],
        outputs=[global_state, form_image],
    )

    def on_click_add_mask(global_state):
        global_state['editing_state'] = 'add_mask'
        return global_state

    enable_add_mask.click(on_click_add_mask,
                          inputs=[global_state],
                          outputs=[global_state])

    def on_click_add_points(global_state):
        global_state['editing_state'] = 'add_points'
        return global_state

    enable_add_points.click(on_click_add_points,
                            inputs=[global_state],
                            outputs=[global_state])

    def on_get_mask(image, global_state):
        # image is a dict with 'image' and 'mask'
        # we only need the mask
        if 'mask' in image:
            mask = get_valid_mask(image['mask'])
            global_state['mask'] = mask
        return global_state

    def on_click_image(global_state, evt: gr.SelectData):
        """Function to handle image click.
        1. If editing_state is add_points:
            1.1 If curr_type_point is start: add start point
            1.2 If curr_type_point is target: add target point
        2. If editing_state is add_mask:
            (Handled by ImageMask component directly)
        """
        if global_state['editing_state'] == 'add_points':
            y, x = evt.index
            if global_state['curr_type_point'] == 'start':
                # Add new point pair
                point_idx = len(global_state['points'])
                global_state['points'][point_idx] = {'start': [y, x], 'target': None}
                global_state['curr_point'] = point_idx
                global_state['curr_type_point'] = 'target'
            else:
                # Add target for current point
                idx = global_state['curr_point']
                global_state['points'][idx]['target'] = [y, x]
                global_state['curr_type_point'] = 'start'
                global_state['curr_point'] = None
            
            image_draw = update_image_draw(
                global_state['images']['image_raw'],
                global_state['points'],
                global_state['mask'],
                global_state['show_mask'],
                global_state,
            )
            return global_state, image_draw
        return global_state, global_state['images']['image_show']

    form_image.select(
        on_click_image,
        inputs=[global_state],
        outputs=[global_state, form_image],
    )

if __name__ == "__main__":
    app.launch(share=args.share, server_name="0.0.0.0" if args.listen else "127.0.0.1")
