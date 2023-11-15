import argparse
import random
import re
import os
from typing import List, Optional, Tuple, Union

import plotly.express as px
import torch
import numpy as np

from datetime import datetime

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.schedulers.scheduling_utils import SchedulerOutput
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin
from PIL import Image
from PIL.PngImagePlugin import PngInfo

class Pipeline(StableDiffusionPipeline):

    @torch.no_grad()
    def __call__(
        self,
        prompt = None,
        height = None,
        width = None,
        num_inference_steps = 50,
        guidance_scale = 7.5,
        negative_prompt = None,
        num_images_per_prompt = 1,
        eta = 0.0,
        generator = None,
        latents = None,
        prompt_embeds = None,
        negative_prompt_embeds = None,
        output_type = "pil",
        return_dict = True,
        callback = None,
        callback_steps = 1,
        cross_attention_kwargs = None,
        guidance_rescale = 0.0,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
            guidance_rescale (`float`, *optional*, defaults to 0.7):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
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
        first_latents = latents

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        noise = []
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                noise.append(latents)
                # call the callback, if provided
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

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept), noise, first_latents

class InterpolateSolver(DPMSolverMultistepScheduler):
    '''

    DPMSolverMultistepScheduler {
      "_class_name": "DPMSolverMultistepScheduler",
      "_diffusers_version": "0.19.3",
      "algorithm_type": "dpmsolver++",
      "beta_end": 0.012,
      "beta_schedule": "scaled_linear",
      "beta_start": 0.00085,
      "clip_sample": false,
      "dynamic_thresholding_ratio": 0.995,
      "lambda_min_clipped": -Infinity,
      "lower_order_final": true,
      "num_train_timesteps": 1000,
      "prediction_type": "epsilon",
      "sample_max_value": 1.0,
      "set_alpha_to_one": false,
      "skip_prk_steps": true,
      "solver_order": 2,
      "solver_type": "midpoint",
      "steps_offset": 1,
      "thresholding": false,
      "timestep_spacing": "linspace",
      "trained_betas": null,
      "use_karras_sigmas": false,
      "variance_type": null
    }

    '''
    def __init__(self):
        self.current_steps = 0
        super().__init__(
            num_train_timesteps = 1000,
            beta_start = 0.0001,
            beta_end = 0.012,
            beta_schedule = "scaled_linear",
            trained_betas = None,
            solver_order = 2,
            prediction_type = "epsilon",
            thresholding = False,
            dynamic_thresholding_ratio = 0.995,
            sample_max_value = 1.0,
            algorithm_type = "dpmsolver++",
            solver_type = "midpoint",
            lower_order_final = True,
            use_karras_sigmas = False,
            lambda_min_clipped = -float("inf"),
            variance_type = None,
            timestep_spacing = "linspace",
            steps_offset = 1,
        )

    def reset(self, noise):
        self.current_steps = 0
        self.noise = noise

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        generator=None,
        return_dict: bool = True,
    ) -> Union[SchedulerOutput, Tuple]:
        """
        Step function propagating the sample with the multistep DPM-Solver.

        Args:
            model_output (`torch.FloatTensor`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.
            return_dict (`bool`): option for returning tuple rather than SchedulerOutput class

        Returns:
            [`~scheduling_utils.SchedulerOutput`] or `tuple`: [`~scheduling_utils.SchedulerOutput`] if `return_dict` is
            True, otherwise a `tuple`. When returning a tuple, the first element is the sample tensor.

        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(self.timesteps.device)
        step_index = (self.timesteps == timestep).nonzero()
        if len(step_index) == 0:
            step_index = len(self.timesteps) - 1
        else:
            step_index = step_index.item()
        prev_timestep = 0 if step_index == len(self.timesteps) - 1 else self.timesteps[step_index + 1]
        lower_order_final = (
            (step_index == len(self.timesteps) - 1) and self.config.lower_order_final and len(self.timesteps) < 15
        )
        lower_order_second = (
            (step_index == len(self.timesteps) - 2) and self.config.lower_order_final and len(self.timesteps) < 15
        )

        model_output = self.convert_model_output(model_output, timestep, sample)
        for i in range(self.config.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
        self.model_outputs[-1] = model_output
        if self.config.algorithm_type in ["sde-dpmsolver", "sde-dpmsolver++"]:
            noise = randn_tensor(
                model_output.shape, generator=generator, device=model_output.device, dtype=model_output.dtype
            )
        else:
            noise = None

        if self.config.solver_order == 1 or self.lower_order_nums < 1 or lower_order_final:
            prev_sample = self.dpm_solver_first_order_update(
                model_output, timestep, prev_timestep, sample, noise=noise
            )
        elif self.config.solver_order == 2 or self.lower_order_nums < 2 or lower_order_second:
            timestep_list = [self.timesteps[step_index - 1], timestep]
            prev_sample = self.multistep_dpm_solver_second_order_update(
                self.model_outputs, timestep_list, prev_timestep, sample, noise=noise
            )
        else:
            timestep_list = [self.timesteps[step_index - 2], self.timesteps[step_index - 1], timestep]
            prev_sample = self.multistep_dpm_solver_third_order_update(
                self.model_outputs, timestep_list, prev_timestep, sample
            )

        if self.lower_order_nums < self.config.solver_order:
            self.lower_order_nums += 1


        prev_sample = self.noise[self.current_steps]
        self.current_steps += 1
        if not return_dict:
            return (prev_sample,)

        return SchedulerOutput(prev_sample=prev_sample)


def prompt_embed(pipe, prompt):
    prompt_embeds = pipe._encode_prompt(
        prompt,
        device='cuda',
        num_images_per_prompt=1,
        do_classifier_free_guidance=True, # should be true if guidance_scale > 1.0
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        lora_scale=None,
    )
    return prompt_embeds[1] # the negative prompt embed is concatenated on top of positive, here we return just the positive ones

def prepare_latents(pipe, generator, prompt_embeds):
    num_channels_latents = pipe.unet.config.in_channels
    latents = pipe.prepare_latents(
        1, #batch_size * num_images_per_prompt,
        num_channels_latents,
        pipe.unet.config.sample_size * pipe.vae_scale_factor, # height
        pipe.unet.config.sample_size * pipe.vae_scale_factor, # width
        prompt_embeds.dtype,
        pipe._execution_device,
        generator,
    )
    return latents

def slerp(p0, p1, t):
    """Spherical Linear Interpolation between two points."""
    shape = np.copy(p0.shape)
    p0 = np.reshape(p0, [-1])
    p1 = np.reshape(p1, [-1])
    omega = np.arccos(np.dot(p0 / np.linalg.norm(p0), p1 / np.linalg.norm(p1)))
    sin_omega = np.sin(omega)
    
    if sin_omega == 0:
        return (1.0 - t) * p0 + t * p1
    
    rotated = np.sin((1.0 - t) * omega) / sin_omega * p0 + np.sin(t * omega) / sin_omega * p1
    return np.reshape(rotated, shape)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_image', required=True, type=str)
    parser.add_argument('--target_image', required=True, type=str)
    parser.add_argument('--steps', default=25, type=int)
    parser.add_argument('--nframes', default=100, type=int)
    parser.add_argument('--destination', default='./prompt-interp', type=str)
    args = parser.parse_args()

    if not os.path.exists(args.destination):
        os.makedirs(args.destination)

    pipe = Pipeline.from_pretrained(
        'stabilityai/stable-diffusion-2-base',
        torch_dtype=torch.float16,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    print(pipe.components.keys())

    start_image = Image.open(args.start_image)
    metadata = start_image.info
    start_seed = float(metadata['seed'])
    start_prompt = metadata['prompt']
    print(start_prompt)
    print(start_seed)

    target_image = Image.open(args.target_image)
    metadata = target_image.info
    target_seed = float(metadata['seed'])
    target_prompt = metadata['prompt'] 
    print(target_prompt)
    print(target_seed)

    # PROMPT INTERPOLATION
    start_prompt_embeds = prompt_embed(pipe, start_prompt)
    target_prompt_embeds = prompt_embed(pipe, target_prompt)
    prompt_interpolations = np.linspace(
        start_prompt_embeds.cpu().detach().numpy(),
        target_prompt_embeds.cpu().detach().numpy(),
        args.nframes
    )[1:]

    '''
    # PROMPT SLERP
    interpolated_points = []
    start_prompt_embeds = start_prompt_embeds.cpu().detach().numpy()
    target_prompt_embeds = target_prompt_embeds.cpu().detach().numpy()
    for i in range(args.nframes):
        t = i / (args.nframes - 1)  # Interpolation parameter between 0 and 1
        interpolated = slerp(start_prompt_embeds, target_prompt_embeds, t)
        interpolated_points.append(interpolated)
    prompt_interpolations = np.array(interpolated_points)
    start_prompt_embeds = torch.from_numpy(start_prompt_embeds).to('cuda')
    target_prompt_embeds = torch.from_numpy(target_prompt_embeds).to('cuda')
    '''
    # START IMAGE
    start_generator = torch.manual_seed(start_seed)
    pipe_out, start_noise, start_latents = pipe(
        prompt_embeds=start_prompt_embeds[None],
        generator=start_generator,
        num_inference_steps=args.steps
    )
    start_image = pipe_out.images[0]
    start_image.save(f'{args.destination}/start.png')
    start_std, start_mean = torch.std_mean(start_latents, keepdim=True)
    '''
    # RANDOM LATENT    
    start_generator = torch.manual_seed(start_seed)
    pipe_out, start_noise, start_latents = pipe(
        prompt_embeds=start_prompt_embeds[None],
        generator=start_generator,
        latents=start_latents,
        num_inference_steps=args.steps
    )
    start_image = pipe_out.images[0]
    start_image.save(f'{args.destination}/random_start.png')
    '''
    
    # TARGET IMAGE
    target_generator = torch.manual_seed(target_seed)
    pipe_out, target_noise, target_latents = pipe(
        prompt_embeds=target_prompt_embeds[None],
        generator=target_generator,
        num_inference_steps=args.steps
    )
    target_image = pipe_out.images[0]
    target_image.save(f'{args.destination}/target.png')
    
    
    #########################################

    # LATENT INTERPOLATION 
    latent_interpolations = np.linspace(
        start_latents.cpu().detach().numpy(),
        target_latents.cpu().detach().numpy(),
        args.nframes
    )[1:]

    # LATENT SLERP
    interpolated_points = []
    start_latents = start_latents.cpu().detach().numpy()
    target_latents = target_latents.cpu().detach().numpy()
    for i in range(args.nframes):
        t = i / (args.nframes - 1)  # Interpolation parameter between 0 and 1
        interpolated = slerp(start_latents, target_latents, t)
        interpolated_points.append(interpolated)
    latent_interpolations = np.array(interpolated_points)
    
    # NOISE INTERPOLATION
    noise_interpolations = []
    for s_noise, t_noise in zip(start_noise, target_noise):
        interpolations = np.linspace(
            s_noise.cpu().detach().numpy(),
            t_noise.cpu().detach().numpy(),
            args.nframes
        )[1:]
        noise_interpolations.append(interpolations)
    noise_interpolations = np.concatenate(noise_interpolations, axis=1)

    #################################################

    images = [start_image]
    

    '''
    # RANDOM WALK
    latent_walk = start_latents
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    for i in range(args.nframes):
        start_generator = torch.manual_seed(start_seed)
        latent_walk += 0.01*torch.normal(torch.tile(start_mean, [1,4,64,64]), start_std)
        pipe_out, middle_noise, middle_latents = pipe(
            prompt_embeds=start_prompt_embeds[None],
            generator=start_generator,
            latents=latent_walk,
            num_inference_steps=args.steps
        )
        image = pipe_out.images[0]
        image.save(f'{args.destination}/image_{len(images)}.png')
        images.append(image)
    '''
    # IMAGE-PROMPT INTERPOLATION
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    for i, prompt_embed in enumerate(prompt_interpolations):
        start_generator = torch.manual_seed(start_seed)
        pipe_out, middle_noise, middle_latents = pipe(
            prompt_embeds=torch.from_numpy(prompt_embed).to('cuda')[None],
            generator=start_generator,
            num_inference_steps=args.steps
        )
        image = pipe_out.images[0]
        image.save(f'{args.destination}/image_{len(images)}.png')
        images.append(image)

    # IMAGE-LATENT INTERPOLATION
#    pipe.scheduler = InterpolateSolver.from_config(pipe.scheduler.config)
    for i, (noise, latents) in enumerate(
        zip(noise_interpolations, latent_interpolations)
    ):
#        pipe.scheduler.reset(torch.from_numpy(noise).to('cuda')[:, None])
#        target_generator = torch.manual_seed(target_seed)
        start_generator = torch.manual_seed(start_seed)
        pipe_out, _, _ = pipe(
            prompt_embeds=target_prompt_embeds[None],
            latents=torch.from_numpy(latents).to('cuda'),
            generator=start_generator,
            num_inference_steps=args.steps
        )
        image = pipe_out.images[0]
        image.save(f'{args.destination}/image_{len(images)}.png')
        images.append(image)
    images.append(target_image)

    # SAVE GIFs
    images += images[2:-1][::-1]
    for fps in [2, 4, 8, 16, 32]:
        images[0].save(
            f'{args.destination}/transition_{fps}fps.gif',
            save_all=True,
            append_images=images[1:],
            duration = 1000 // fps,
            loop=0
        )
