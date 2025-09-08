import os
from PIL import Image
import re
import numpy as np
import torch
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from typing import List, Optional, Tuple, Union

def get_frame_files(directory):
    """Gets all frame files from a directory, sorted by frame number."""
    frame_files = [f for f in os.listdir(directory) if f.startswith('frame_') and f.endswith('.png')]
    
    def get_frame_number(filename):
        match = re.search(r'frame_(\d+).png', filename)
        return int(match.group(1)) if match else -1

    return sorted(frame_files, key=get_frame_number)

def create_gif(frame_folder, output_path, bounce=True, duration=100):
    """Creates a GIF from a folder of frames."""
    frames = []
    frame_files = get_frame_files(frame_folder)
    for frame_file in frame_files:
        frame_path = os.path.join(frame_folder, frame_file)
        frames.append(Image.open(frame_path))

    if not frames:
        print("No frames found to create a GIF.")
        return

    if bounce:
        # Create a bouncing effect by appending the reversed frames (excluding start and end)
        append_images = frames[1:] + frames[-2:0:-1]
    else:
        append_images = frames[1:]

    frames[0].save(
        output_path,
        save_all=True,
        append_images=append_images,
        duration=duration,
        loop=0
    )
    print(f"Saved GIF to {output_path}")

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
        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept), first_latents, prompt_embeds