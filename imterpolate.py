import argparse
import os
import torch
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
import re
from typing import List, Optional, Tuple, Union


# --- Copied from imterpolate.py (and diffusers) ---

def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1.0 - guidance_rescale) * noise_cfg
    return noise_cfg

class InterpPipeline(StableDiffusionPipeline):
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[callable] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[dict] = None,
        guidance_rescale: float = 0.0,
    ) -> Union[StableDiffusionPipelineOutput, Tuple, torch.FloatTensor]:
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        first_latents = latents.clone() # Use clone to avoid modification
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)
        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None
        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]
        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()
        if not return_dict:
            return (image, has_nsfw_concept)
        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept), first_latents

def slerp(p0, p1, t):
    """Spherical Linear Interpolation between two points."""
    shape = np.copy(p0.shape)
    p0 = np.reshape(p0, [-1])
    p1 = np.reshape(p1, [-1])
    omega = np.arccos(np.dot(p0 / np.linalg.norm(p0), p1 / np.linalg.norm(p1)))
    sin_omega = np.sin(omega)
    
    if sin_omega == 0:
        return np.reshape((1.0 - t) * p0 + t * p1, shape)
    
    rotated = np.sin((1.0 - t) * omega) / sin_omega * p0 + np.sin(t * omega) / sin_omega * p1
    return np.reshape(rotated, shape)

def interpolate(start, target, nframes, type):
    if type == 'linear':
        interpolations = np.linspace(start, target, nframes)
    elif type == 'slerp':
        interpolated_points = []
        for i in range(nframes):
            t = i / (nframes - 1)
            interpolated = slerp(start, target, t)
            interpolated_points.append(interpolated)
        interpolations = np.array(interpolated_points)
    else:
        raise ValueError(f'Unrecognized interpolation type {type}')
    return interpolations

# --- Main script ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Interpolate between two images using latent space SLERP.")
    parser.add_argument('--start_image', required=True, type=str, help="Path to the starting image.")
    parser.add_argument('--target_image', required=True, type=str, help="Path to the target image.")
    parser.add_argument('--nframes', default=60, type=int, help="Number of frames to generate for the interpolation.")
    parser.add_argument('--destination', default='./latent-interpolation', type=str, help="Base directory to save the output runs.")
    parser.add_argument('--no-gif', action='store_true', help="Do not generate a GIF of the final frames.")
    args = parser.parse_args()

    # --- Create a new numbered subfolder for this run ---
    os.makedirs(args.destination, exist_ok=True)
    existing_runs = [d for d in os.listdir(args.destination) if os.path.isdir(os.path.join(args.destination, d)) and d.isdigit()]
    next_run_number = 0
    if existing_runs:
        next_run_number = max([int(d) for d in existing_runs]) + 1
    run_destination = os.path.join(args.destination, f"{next_run_number:03d}")
    os.makedirs(run_destination)
    print(f"This is run #{next_run_number}. Saving output to: {run_destination}")

    print("Loading model...")
    pipe = InterpPipeline.from_pretrained(
        "Lykon/dreamshaper-7",
    )
    pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
    print("Fusing LoRA weights for performance...")
    pipe.fuse_lora()
    pipe.enable_model_cpu_offload()
    print("Disabling safety checker...")
    pipe.safety_checker = None
    print("Model loaded.")

    # --- Get metadata from images ---
    start_image_pil = Image.open(args.start_image)
    start_metadata = start_image_pil.info
    start_seed = int(float(start_metadata['seed']))
    start_prompt = start_metadata['prompt']
    print(f"Start prompt: {start_prompt}")
    print(f"Start seed: {start_seed}")

    target_image_pil = Image.open(args.target_image)
    target_metadata = target_image_pil.info
    target_seed = int(float(target_metadata['seed']))
    target_prompt = target_metadata['prompt']
    print(f"Target prompt: {target_prompt}")
    print(f"Target seed: {target_seed}")

    # --- Save prompts to a text file ---
    prompt_info = f"Start Prompt: {start_prompt}\n"
    prompt_info += f"Target Prompt: {target_prompt}\n"
    prompts_filepath = os.path.join(run_destination, "prompts.txt")
    with open(prompts_filepath, "w") as f:
        f.write(prompt_info)
    print(f"Saved prompts to {prompts_filepath}")

    # --- Generate and save START image ---
    print("Generating start image...")
    start_generator = torch.manual_seed(start_seed)
    start_prompt_embeds, _ = pipe.encode_prompt(start_prompt, pipe.device, 1, False)
    start_pipe_out, start_latents = pipe(
        prompt=start_prompt,
        generator=start_generator,
        num_inference_steps=16,
        guidance_scale=0.0
    )
    start_image = start_pipe_out.images[0]
    start_savename = f"frame_{0:04d}.png"
    start_savepath = os.path.join(run_destination, start_savename)
    start_image.save(start_savepath)
    print(f"Saved start image to {start_savepath}")

    # --- Generate and save TARGET image ---
    print("Generating target image...")
    target_generator = torch.manual_seed(target_seed)
    target_prompt_embeds, _ = pipe.encode_prompt(target_prompt, pipe.device, 1, False)
    target_pipe_out, target_latents = pipe(
        prompt=target_prompt,
        generator=target_generator,
        num_inference_steps=16,
        guidance_scale=0.0
    )
    target_image = target_pipe_out.images[0]
    target_savename = f"frame_{args.nframes - 1:04d}.png"
    target_savepath = os.path.join(run_destination, target_savename)
    target_image.save(target_savepath)
    print(f"Saved target image to {target_savepath}")

    # --- Interpolate ---
    print("Interpolating latents and prompts...")
    latent_interpolations = interpolate(start_latents.detach().cpu().numpy(), target_latents.detach().cpu().numpy(), args.nframes, 'slerp')
    prompt_embed_interpolations = interpolate(start_prompt_embeds.detach().cpu().numpy(), target_prompt_embeds.detach().cpu().numpy(), args.nframes, 'linear')

    # --- Generate INTERMEDIATE frames ---
    print(f"Generating {args.nframes - 2} intermediate frames...")
    interp_generator = torch.manual_seed(start_seed)
    for i in range(1, args.nframes - 1):
        print(f"  Frame {i+1}/{args.nframes}")
        interp_generator.manual_seed(start_seed)
        pipe_out, _ = pipe(
            prompt_embeds=torch.from_numpy(prompt_embed_interpolations[i]).to(pipe.device, dtype=pipe.dtype),
            latents=torch.from_numpy(latent_interpolations[i]).to(pipe.device, dtype=pipe.dtype),
            generator=interp_generator,
            num_inference_steps=16,
            guidance_scale=0.0
        )
        image = pipe_out.images[0]
        savename = f"frame_{i:04d}.png"
        savepath = os.path.join(run_destination, savename)
        image.save(savepath)

    # --- Create GIF ---
    if not args.no_gif:
        print("Saving GIF...")
        frames = []
        for i in range(args.nframes):
            frame_path = os.path.join(run_destination, f"frame_{i:04d}.png")
            frames.append(Image.open(frame_path))
        
        gif_path = os.path.join(run_destination, 'transition.gif')
        # Create a bouncing effect by appending the reversed frames (excluding start and end)
        bouncing_frames = frames[1:] + frames[-2:0:-1]
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=bouncing_frames,
            duration=100, # 10 fps
            loop=0
        )
        print(f"Saved GIF to {gif_path}")

    print(f"Done! Output saved in {run_destination}")

