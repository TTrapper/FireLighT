import argparse
import os
import re
import shutil
import subprocess
import tempfile
from utils import get_frame_files, create_gif

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create a new interpolation by bridging two existing frames.")
    parser.add_argument('--start_image', required=True, type=str, help="Path to the starting frame (e.g., /path/to/run1/frame_0010.png).")
    parser.add_argument('--target_image', required=True, type=str, help="Path to the target frame (e.g., /path/to/run2/frame_0025.png).")
    parser.add_argument('--nframes_transition', type=int, default=30, help="Number of new frames to insert between start and target.")
    parser.add_argument('--destination', default='./latent-interpolation-stitched', type=str, help="Base directory to save the output runs.")
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

    # --- Path and frame number parsing ---
    start_folder = os.path.dirname(args.start_image)
    start_filename = os.path.basename(args.start_image)
    start_frame_match = re.search(r'frame_(\d+).png', start_filename)
    if not start_frame_match:
        parser.error("Start image filename is not in the format frame_xxxx.png")
    start_frame_num = int(start_frame_match.group(1))

    target_folder = os.path.dirname(args.target_image)
    target_filename = os.path.basename(args.target_image)
    target_frame_match = re.search(r'frame_(\d+).png', target_filename)
    if not target_frame_match:
        parser.error("Target image filename is not in the format frame_xxxx.png")
    target_frame_num = int(target_frame_match.group(1))

    # --- Stitching Logic ---
    tensors_destination = os.path.join(run_destination, "tensors")
    os.makedirs(tensors_destination, exist_ok=True)
    current_frame_idx = 0

    # 1. Copy frames up to and including start_image
    print(f"Copying {start_frame_num + 1} frames from {start_folder}...")
    for i in range(start_frame_num + 1):
        shutil.copy(os.path.join(start_folder, f"frame_{i:04d}.png"), os.path.join(run_destination, f"frame_{current_frame_idx:04d}.png"))
        shutil.copy(os.path.join(start_folder, "tensors", f"frame_{i:04d}_latent.pt"), os.path.join(tensors_destination, f"frame_{current_frame_idx:04d}_latent.pt"))
        shutil.copy(os.path.join(start_folder, "tensors", f"frame_{i:04d}_prompt_embed.pt"), os.path.join(tensors_destination, f"frame_{current_frame_idx:04d}_prompt_embed.pt"))
        current_frame_idx += 1

    # 2. Generate and copy transition frames
    print(f"Generating {args.nframes_transition} new transition frames...")
    with tempfile.TemporaryDirectory() as temp_dir:
        transition_dest = os.path.join(temp_dir, "transition")
        start_latent_path = os.path.join(start_folder, "tensors", f"frame_{start_frame_num:04d}_latent.pt")
        start_prompt_path = os.path.join(start_folder, "tensors", f"frame_{start_frame_num:04d}_prompt_embed.pt")
        target_latent_path = os.path.join(target_folder, "tensors", f"frame_{target_frame_num:04d}_latent.pt")
        target_prompt_path = os.path.join(target_folder, "tensors", f"frame_{target_frame_num:04d}_prompt_embed.pt")

        subprocess.run([
            'python', 'imterpolate.py',
            '--start_latent', start_latent_path,
            '--start_prompt_embed', start_prompt_path,
            '--target_latent', target_latent_path,
            '--target_prompt_embed', target_prompt_path,
            '--nframes', str(args.nframes_transition + 2),
            '--destination', transition_dest,
            '--no-gif',
            '--only_intermediate'
        ])

        transition_run_folder = os.path.join(transition_dest, "000")
        transition_frames = get_frame_files(transition_run_folder)
        for frame_file in transition_frames:
            frame_num_match = re.search(r'frame_(\d+).png', frame_file)
            frame_num = int(frame_num_match.group(1))
            shutil.copy(os.path.join(transition_run_folder, frame_file), os.path.join(run_destination, f"frame_{current_frame_idx:04d}.png"))
            shutil.copy(os.path.join(transition_run_folder, "tensors", f"frame_{frame_num:04d}_latent.pt"), os.path.join(tensors_destination, f"frame_{current_frame_idx:04d}_latent.pt"))
            shutil.copy(os.path.join(transition_run_folder, "tensors", f"frame_{frame_num:04d}_prompt_embed.pt"), os.path.join(tensors_destination, f"frame_{current_frame_idx:04d}_prompt_embed.pt"))
            current_frame_idx += 1

    # 3. Copy frames from target_image to the end
    target_all_frames = get_frame_files(target_folder)
    num_target_frames = len(target_all_frames)
    print(f"Copying {num_target_frames - target_frame_num} frames from {target_folder}...")
    for i in range(target_frame_num, num_target_frames):
        shutil.copy(os.path.join(target_folder, f"frame_{i:04d}.png"), os.path.join(run_destination, f"frame_{current_frame_idx:04d}.png"))
        shutil.copy(os.path.join(target_folder, "tensors", f"frame_{i:04d}_latent.pt"), os.path.join(tensors_destination, f"frame_{current_frame_idx:04d}_latent.pt"))
        shutil.copy(os.path.join(target_folder, "tensors", f"frame_{i:04d}_prompt_embed.pt"), os.path.join(tensors_destination, f"frame_{current_frame_idx:04d}_prompt_embed.pt"))
        current_frame_idx += 1

    # --- Create GIF ---
    if not args.no_gif:
        gif_path = os.path.join(run_destination, 'transition.gif')
        create_gif(run_destination, gif_path)

    print(f"Done! Stitched output saved in {run_destination}")