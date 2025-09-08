from flask import Flask, render_template, request, send_file, jsonify
import os
import random
import base64
import shutil
import threading
import time
import re
from datetime import datetime
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import torch
from diffusers import AutoPipelineForText2Image
from train import load_data

from diffusers import AmusedPipeline

app = Flask(__name__)

# --- Configuration ---
IMAGE_DIR = "./images"
GOOD_DIR = "./good_images"
BAD_DIR = "./bad_images"
MEH_DIR = "./meh_images"
MAX_QUEUE_SIZE = 20
GENERATION_INTERVAL_SECONDS = 2  # Time to wait between generating images

# --- Setup Directories ---
for d in [IMAGE_DIR, GOOD_DIR, BAD_DIR, MEH_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

# --- Load Model (once on startup) ---
print("Loading model...")
pipe = AutoPipelineForText2Image.from_pretrained(
     "Lykon/dreamshaper-7"
)

pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
#pipe.fuse_lora()
pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
#pipe.to("cuda")
pipe.safety_checker = None
print("Model loaded.")

# --- Prompt Generation Data (from album_cover.py) ---
names = list(load_data('./ftl-fixed.csv')['ftl'])
artstyles = [
    'grainy', 'spooky', 'beautiful', 'epic', 'pixelart', 'horror', 'colorful',
    'gloomey', 'haunting', 'hellish', 'outer space', 'alien', 'retro sci-fi',
    'stylized', 'fantasy', 'textured', 'vibrant', 'echo', 'quiet',
    'reverberating', 'trill', 'meditative', 'clear', 'fuzz', 'murky',
    'organized', 'organic', 'futurism', 'aerospace', 'galactic',
]
mediums = [
    'painting', 'photograph', 'drawing', 'watercolor', 'digital drawing',
    'piece', 'artwork', '',
]

# --- Background Image Generation Thread ---
def generate_images_continuously():
    """A background thread that continuously generates images."""
    while True:
        try:
            num_images = len(os.listdir(IMAGE_DIR))
            if num_images < MAX_QUEUE_SIZE:
                print(f"Queue has {num_images}/{MAX_QUEUE_SIZE} images. Generating a new one.")

                # Create prompt
                name = random.choice(names)
                seed = int(datetime.now().timestamp())
                medium = random.choice(mediums)
                styles = ' '.join([medium] + [random.choice(artstyles) for _ in range(4)])
                prompt = f'{styles} piece called {name}'

                # Generate image
                generator = torch.manual_seed(seed)
                image = pipe(
                    prompt,
                    generator=generator,
                    num_inference_steps=16,
                    guidance_scale=0.0
                ).images[0]

                # Save image with metadata
                savename = re.sub(r'[^\w.-]', '_', prompt)
                savename = f'{savename}.{str(seed)}'
                savepath = os.path.join(IMAGE_DIR, f'{savename}.png')

                metadata = PngInfo()
                metadata.add_text("name", f"{name}")
                metadata.add_text("prompt", prompt)
                metadata.add_text("seed", str(seed))
                metadata.add_text("steps", "1")
                image.save(savepath, pnginfo=metadata)
                print(f"Saved: {savepath}")

        except Exception as e:
            print(f"Error in generation thread: {e}")

        time.sleep(GENERATION_INTERVAL_SECONDS)

# --- Flask Routes ---
@app.route("/")
def home():
    """Serves the main art display page."""
    return render_template('art.html')

@app.get('/api/image_queue')
def get_image_queue():
    """Returns a sorted list of images in the queue."""
    try:
        image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
        # Sort by creation time (oldest first)
        sorted_files = sorted(
            image_files,
            key=lambda f: os.path.getctime(os.path.join(IMAGE_DIR, f))
        )
        return jsonify([os.path.join(IMAGE_DIR, f) for f in sorted_files])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.post('/api/rate_image')
def rate_image():
    """Moves an image to the good, bad, or meh folder."""
    img_filepath = request.form.get('filepath')
    rating = request.form.get('rating')

    if not img_filepath or not rating:
        return jsonify({'message': 'Missing filepath or rating'}), 400
    if not os.path.exists(img_filepath):
        return jsonify({'message': 'Image not found'}), 404

    img_filename = os.path.basename(img_filepath)
    dest_dir = None
    if rating == 'good':
        dest_dir = GOOD_DIR
    elif rating == 'bad':
        dest_dir = BAD_DIR
    elif rating == 'meh':
        dest_dir = MEH_DIR
    else:
        return jsonify({'message': 'Invalid rating'}), 400

    try:
        shutil.move(img_filepath, os.path.join(dest_dir, img_filename))
        return jsonify({'message': f'Image moved to {rating} folder'})
    except Exception as e:
        return jsonify({'message': f'Error moving file: {e}'}), 500

@app.get('/get_image_data')
def get_image_data():
    """Gets an image's base64 data and metadata."""
    img_path = request.args.get('filepath')
    if not img_path or not os.path.exists(img_path):
        return jsonify({"error": "File not found"}), 404

    try:
        with open(img_path, 'rb') as img_file:
            img_data = base64.b64encode(img_file.read()).decode('utf8')

        img = Image.open(img_path)
        img_metadata = img.info
        name = img_metadata.get('name', 'N/A')
        prompt = img_metadata.get('prompt', 'N/A')
        img.close()

        return jsonify({
            'name': name,
            'prompt': prompt,
            'image_data': img_data
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Start the background thread
    generator_thread = threading.Thread(target=generate_images_continuously, daemon=True)
    generator_thread.start()
    # Start the Flask app
    app.run(host='0.0.0.0', port=5001)
