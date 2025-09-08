import argparse
import os
import torch
from PIL import Image
from utils import create_gif, interpolate, InterpPipeline


# --- Main script ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Interpolate between two images or tensors.")
    parser.add_argument('--start_image', type=str, help="Path to the starting image.")
    parser.add_argument('--target_image', type=str, help="Path to the target image.")
    parser.add_argument('--start_latent', type=str, help="Path to the starting latent tensor (.pt file).")
    parser.add_argument('--start_prompt_embed', type=str, help="Path to the starting prompt embed tensor (.pt file).")
    parser.add_argument('--target_latent', type=str, help="Path to the target latent tensor (.pt file).")
    parser.add_argument('--target_prompt_embed', type=str, help="Path to the target prompt embed tensor (.pt file).")
    parser.add_argument('--nframes', default=60, type=int, help="Number of frames to generate for the interpolation.")
    parser.add_argument('--destination', default='./latent-interpolation', type=str, help="Base directory to save the output runs.")
    parser.add_argument('--no-gif', action='store_true', help="Do not generate a GIF of the final frames.")
    parser.add_argument('--only_intermediate', action='store_true', help="Only generate intermediate frames, excluding endpoints.")
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

    tensors_destination = os.path.join(run_destination, "tensors")
    os.makedirs(tensors_destination)

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

    if args.start_image and args.target_image:
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

        # --- Generate START latents and prompt embeds ---
        start_generator = torch.manual_seed(start_seed)
        start_prompt_embeds, _ = pipe.encode_prompt(start_prompt, pipe.device, 1, False)
        _, start_latents = pipe(
            prompt=start_prompt,
            generator=start_generator,
            num_inference_steps=16,
            guidance_scale=0.0
        )

        # --- Generate TARGET latents and prompt embeds ---
        target_generator = torch.manual_seed(target_seed)
        target_prompt_embeds, _ = pipe.encode_prompt(target_prompt, pipe.device, 1, False)
        _, target_latents = pipe(
            prompt=target_prompt,
            generator=target_generator,
            num_inference_steps=16,
            guidance_scale=0.0
        )

    elif args.start_latent and args.start_prompt_embed and args.target_latent and args.target_prompt_embed:
        start_latents = torch.load(args.start_latent)
        start_prompt_embeds = torch.load(args.start_prompt_embed)
        target_latents = torch.load(args.target_latent)
        target_prompt_embeds = torch.load(args.target_prompt_embed)
        start_seed = int(torch.initial_seed()) # Use a random seed for the generator
    else:
        parser.error("You must provide either (--start_image and --target_image) or (--start_latent, --start_prompt_embed, --target_latent, and --target_prompt_embed).")

    # --- Interpolate ---
    print("Interpolating latents and prompts...")
    latent_interpolations = interpolate(start_latents.detach().cpu().numpy(), target_latents.detach().cpu().numpy(), args.nframes, 'slerp')
    prompt_embed_interpolations = interpolate(start_prompt_embeds.detach().cpu().numpy(), target_prompt_embeds.detach().cpu().numpy(), args.nframes, 'linear')

    # --- Generate ALL frames ---
    print(f"Generating {args.nframes} frames...")
    interp_generator = torch.manual_seed(start_seed)
    for i in range(args.nframes):
        if args.only_intermediate and (i == 0 or i == args.nframes - 1):
            continue
        print(f"  Frame {i+1}/{args.nframes}")
        interp_generator.manual_seed(start_seed)
        
        latent_tensor = torch.from_numpy(latent_interpolations[i]).to(pipe.device, dtype=pipe.dtype)
        prompt_embed_tensor = torch.from_numpy(prompt_embed_interpolations[i]).to(pipe.device, dtype=pipe.dtype)

        pipe_out, _ = pipe(
            prompt_embeds=prompt_embed_tensor,
            latents=latent_tensor,
            generator=interp_generator,
            num_inference_steps=16,
            guidance_scale=0.0
        )
        image = pipe_out.images[0]
        savename = f"frame_{i:04d}.png"
        savepath = os.path.join(run_destination, savename)
        image.save(savepath)
        torch.save(latent_tensor, os.path.join(tensors_destination, f"frame_{i:04d}_latent.pt") )
        torch.save(prompt_embed_tensor, os.path.join(tensors_destination, f"frame_{i:04d}_prompt_embed.pt") )

    # --- Create GIF ---
    if not args.no_gif:
        print("Saving GIF...")
        gif_path = os.path.join(run_destination, 'transition.gif')
        create_gif(run_destination, gif_path)

    print(f"Done! Output saved in {run_destination}")