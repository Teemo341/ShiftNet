import share

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random
import copy

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from shiftdm.model import create_model, load_state_dict
from shiftdm.ddim_hacked import DDIMSampler


preprocessor = None
# Auto-detect device: cuda, mps, or cpu
if torch.cuda.is_available():
    DEVICE = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'

model_name = 'shift_sd15_canny'
model = create_model(f'./models/shiftdm/{model_name}.yaml').cpu()
model.load_state_dict(load_state_dict('./models/sd/v1-5-pruned.ckpt', location=DEVICE), strict=False)
# model.load_state_dict(load_state_dict(f'./models/shiftdm/{model_name}.pth', location=DEVICE), strict=False)
model = model.to(DEVICE)
ddim_sampler = DDIMSampler(model)


def process(det, input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, shift_strength, scale, seed, eta, low_threshold, high_threshold):
    global preprocessor

    if det == 'Canny':
        if not isinstance(preprocessor, CannyDetector):
            preprocessor = CannyDetector()

    with torch.no_grad():
        input_image = HWC3(input_image)

        if det == 'None':
            detected_map = input_image.copy()
        else:
            detected_map = preprocessor(resize_image(input_image, detect_resolution), low_threshold, high_threshold)
            detected_map = HWC3(detected_map)

        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

        shift = torch.from_numpy(detected_map.copy()).float().to(DEVICE) / 255.0
        shift = torch.stack([shift for _ in range(num_samples)], dim=0)
        shift = einops.rearrange(shift, 'b h w c -> b c h w').clone()
        shift_rec = copy.deepcopy(shift)
        shift_rec = model.decode_first_stage(model.get_shift_stage_encoding(model.encode_shift_stage({'canny':shift_rec})))
        shift_rec = (einops.rearrange(shift_rec, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)[0]
        shift = {'canny': shift}

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if share.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"shift": shift, "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        if guess_mode:
            un_cond = {"c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        else:
            un_cond = {"shift": shift, "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if share.save_memory:
            model.low_vram_shift(is_diffusing=True)

        # model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
        # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        model.shift_stage_scale = shift_strength

        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if share.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return detected_map, shift_rec, results


block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## Control Stable Diffusion with Canny Edges")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="numpy")
            prompt = gr.Textbox(label="Prompt")
            run_button = gr.Button(value="Run")
            num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
            seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, value=12345)
            det = gr.Radio(choices=["Canny", "None"], type="value", value="Canny", label="Preprocessor")
            with gr.Accordion("Advanced options", open=False):
                low_threshold = gr.Slider(label="Canny low threshold", minimum=1, maximum=255, value=100, step=1)
                high_threshold = gr.Slider(label="Canny high threshold", minimum=1, maximum=255, value=200, step=1)
                image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
                shift_strength = gr.Slider(label="Shift Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                guess_mode = gr.Checkbox(label='Guess Mode', value=False)
                detect_resolution = gr.Slider(label="Preprocessor Resolution", minimum=128, maximum=1024, value=512, step=1)
                ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
                scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
                eta = gr.Slider(label="DDIM ETA", minimum=0.0, maximum=1.0, value=1.0, step=0.01)
                a_prompt = gr.Textbox(label="Added Prompt", value='best quality')
                n_prompt = gr.Textbox(label="Negative Prompt", value='lowres, bad anatomy, bad hands, cropped, worst quality')
        with gr.Column():
            with gr.Row():
                    detected_image = gr.Image(label="Detected Image", type="numpy")
                    shift_image = gr.Image(label="Shift Image", type="numpy")
            with gr.Row():
                result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery")
    ips = [det, input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, shift_strength, scale, seed, eta, low_threshold, high_threshold]
    run_button.click(fn=process, inputs=ips, outputs=(detected_image, shift_image, result_gallery))


block.launch(server_port=8388)
# block.launch()
