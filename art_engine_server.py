import os
import uuid
import json
import random
import threading
import time
from datetime import datetime
from flask import Flask, jsonify, request, send_from_directory, render_template
from flask_cors import CORS
import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image
import numpy as np
import pandas as pd
from utils import interpolate, InterpPipeline


# --- Configuration ---
DATA_DIR = "./data"
NODES_DIR = os.path.join(DATA_DIR, "nodes")
TIMELINES_DIR = os.path.join(DATA_DIR, "timelines")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
TENSORS_DIR = os.path.join(DATA_DIR, "tensors")
MAX_QUEUE_SIZE = 20
GENERATION_INTERVAL_SECONDS = 5  # Time to wait between generating images
MODEL_NAME = "Lykon/dreamshaper-7"
LORA_NAME = "latent-consistency/lcm-lora-sdv1-5"
PIPELINE = "AutoPipelineForText2Image"


# --- Setup Directories ---
for d in [NODES_DIR, TIMELINES_DIR, IMAGES_DIR, TENSORS_DIR]:
    os.makedirs(d, exist_ok=True)

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app) # Allow cross-origin requests for the frontend

# --- Load Model (once on startup) ---
print("Loading diffusion model...")
pipe = InterpPipeline.from_pretrained(
    MODEL_NAME,
)
pipe.load_lora_weights(LORA_NAME)
pipe.fuse_lora() # Fuse LoRA weights for performance
pipe.enable_model_cpu_offload()
pipe.safety_checker = None
print("Model loaded.")

# --- Prompt Generation Data (from artserver.py) ---
PROMPT_DATA_DIR = "./config"
PROMPT_DATA_FILE = os.path.join(PROMPT_DATA_DIR, "prompt_data.json")

# Ensure the config directory exists
os.makedirs(PROMPT_DATA_DIR, exist_ok=True)

def load_prompt_data():
    if os.path.exists(PROMPT_DATA_FILE):
        with open(PROMPT_DATA_FILE, 'r') as f:
            return json.load(f)
    else:
        return {
                'names': ['banana', 'apple', 'pear'],
                'artstyles': ['happy', 'colorful', 'dark'],
                'mediums': ['photograph', 'watercolor', 'drawing']
               }

def save_prompt_data(data):
    with open(PROMPT_DATA_FILE, 'w') as f:
        json.dump(data, f, indent=2)

prompt_data = load_prompt_data()
names = prompt_data["names"]
artstyles = prompt_data["artstyles"]
mediums = prompt_data["mediums"]

@app.route('/v1/prompt_data', methods=['GET'])
def get_prompt_data():
    return jsonify(prompt_data)

@app.route('/v1/prompt_data', methods=['POST'])
def update_prompt_data():
    global prompt_data, names, artstyles, mediums
    data = request.json
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid data format. Expected JSON object."}), 400

    # Validate and update each list if present in the request
    updated = False
    if "names" in data and isinstance(data["names"], list):
        prompt_data["names"] = data["names"]
        names = data["names"]
        updated = True
    if "artstyles" in data and isinstance(data["artstyles"], list):
        prompt_data["artstyles"] = data["artstyles"]
        artstyles = data["artstyles"]
        updated = True
    if "mediums" in data and isinstance(data["mediums"], list):
        prompt_data["mediums"] = data["mediums"]
        mediums = data["mediums"]
        updated = True

    if updated:
        save_prompt_data(prompt_data)
        return jsonify({"message": "Prompt data updated successfully", "data": prompt_data}), 200
    else:
        return jsonify({"error": "No valid prompt data (names, artstyles, or mediums) provided for update."}), 400

# --- Core Logic ---

def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    """Spherical linear interpolation."""
    v0 = v0.flatten().numpy()
    v1 = v1.flatten().numpy()
    dot = np.sum(v0 * v1)
    if np.abs(dot) > DOT_THRESHOLD:
        return (1 - t) * v0 + t * v1
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta_t = theta_0 * t
    sin_theta_t = np.sin(theta_t)
    s0 = np.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    return (s0 * v0 + s1 * v1).reshape(1, 4, 64, 64) # Reshape back to latent dimensions

def generate_new_art_node(prompt_text=None, seed=None):
    """Generates a new image, tensors, and metadata file."""
    node_id = str(uuid.uuid4())
    print(f"Generating new ArtNode: {node_id}")

    if not prompt_text:
        name = random.choice(names)
        medium = random.choice(mediums)
        # Combine a medium with 4 random styles
        styles = ' '.join([medium] + [random.choice(artstyles) for _ in range(4)])
        prompt_text = f'{styles} piece called {name}'
    
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    generator = torch.manual_seed(seed)
    
    with torch.no_grad():
        pipe_out, latents, prompt_embeds = pipe(
            prompt=prompt_text,
            generator=generator,
            num_inference_steps=16,
            guidance_scale=0.0
        )
        image = pipe_out.images[0]

    # Define paths
    image_path = os.path.join(IMAGES_DIR, f"{node_id}.png")
    latent_path = os.path.join(TENSORS_DIR, f"{node_id}_latent.pt")
    prompt_embed_path = os.path.join(TENSORS_DIR, f"{node_id}_prompt_embed.pt")
    node_meta_path = os.path.join(NODES_DIR, f"{node_id}.json")

    # Save assets
    image.save(image_path)
    torch.save(latents, latent_path)
    torch.save(prompt_embeds, prompt_embed_path)

    # Create and save metadata
    art_node = {
        "id": node_id,
        "image_path": f"/images/{node_id}.png",
        "latent_path": latent_path,
        "prompt_embed_path": prompt_embed_path,
        "model_info": {
            "name": MODEL_NAME,
            "lora": LORA_NAME,
            "pipeline": PIPELINE
        },
        "prompt_text": prompt_text,
        "seed": seed,
        "created_at": datetime.utcnow().isoformat(),
        "parent_nodes": [],
        "interp_alpha": None,
        "rating": None,
    }
    with open(node_meta_path, 'w') as f:
        json.dump(art_node, f, indent=2)

    return art_node

def generate_image_from_tensors(latents, prompt_embeds, seed=0):
    """Generates an image from existing latent and prompt tensors."""
    generator = torch.manual_seed(seed)
    with torch.no_grad():
        pipe_out, _, _ = pipe(
            prompt_embeds=prompt_embeds.to(pipe.device),
            latents=latents.to(pipe.device),
            generator=generator,
            num_inference_steps=16,
            guidance_scale=0.0
        )
        image = pipe_out.images[0]
    return image
    



# --- Global State for Generation ---
GENERATION_STATUS = {
    "is_generating": False,
    "total_images": 0,
    "completed_images": 0,
}
GENERATION_LOCK = threading.Lock()

@app.route('/v1/status', methods=['GET'])
def get_status():
    with GENERATION_LOCK:
        return jsonify(GENERATION_STATUS)

def generate_n_images(count):
    """Generates a given number of new art nodes in a thread."""
    global GENERATION_STATUS
    with GENERATION_LOCK:
        if GENERATION_STATUS["is_generating"]:
            print("Generation is already in progress.")
            return
        GENERATION_STATUS["is_generating"] = True
        GENERATION_STATUS["total_images"] = count
        GENERATION_STATUS["completed_images"] = 0

    print(f"Starting background generation of {count} images.")
    for i in range(count):
        try:
            print(f"Generating image {i+1}/{count}...")
            generate_new_art_node()
            with GENERATION_LOCK:
                GENERATION_STATUS["completed_images"] = i + 1
        except Exception as e:
            print(f"Error during batch generation: {e}")
            # Optionally break or continue
            break # Stop generation on error

    with GENERATION_LOCK:
        GENERATION_STATUS["is_generating"] = False
    print(f"Finished generating {count} images.")



# --- API Endpoints ---

@app.route('/')
def index():
    """Serves the main UI."""
    return render_template('art_engine_ui.html')

@app.route('/v1/generate', methods=['POST'])
def trigger_generation():
    count = request.json.get('num_images', 1)
    if not isinstance(count, int) or count <= 0 or count > 100:
         return jsonify({"error": "Invalid 'count'. Must be an integer between 1 and 100."}), 400

    # Run generation in a background thread to not block the request
    thread = threading.Thread(target=generate_n_images, args=(count,))
    thread.daemon = True
    thread.start()

    return jsonify({"message": f"Started generating {count} images in the background."}), 202

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(IMAGES_DIR, filename)

@app.route('/v1/nodes', methods=['GET'])
def get_all_nodes():
    nodes = []
    for fname in os.listdir(NODES_DIR):
        if fname.endswith('.json'):
            with open(os.path.join(NODES_DIR, fname), 'r') as f:
                node_data = json.load(f)
                nodes.append(node_data)
    
    # Sort nodes by 'created_at' timestamp, most recent first
    nodes.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    
    return jsonify(nodes)

@app.route('/v1/nodes/<node_id>/rate', methods=['POST'])
def rate_node(node_id):
    node_path = os.path.join(NODES_DIR, f"{node_id}.json")
    if not os.path.exists(node_path):
        return jsonify({"error": "Node not found"}), 404
    
    with open(node_path, 'r') as f:
        node_data = json.load(f)
        
    node_data['rating'] = request.json.get('rating')
    
    with open(node_path, 'w') as f:
        json.dump(node_data, f, indent=2)
        
    return jsonify(node_data)



@app.route('/v1/timelines/<timeline_id>/interpolate', methods=['POST'])
def interpolate_in_timeline(timeline_id):
    timeline_path = os.path.join(TIMELINES_DIR, f"{timeline_id}.json")
    if not os.path.exists(timeline_path):
        return jsonify({"error": "Timeline not found"}), 404
        
    data = request.json
    start_node_id = data.get('start_node_id')
    end_node_id = data.get('end_node_id')
    steps = int(data.get('steps', 10)) # Default to a smaller number for in-place

    with open(timeline_path, 'r') as f:
        timeline = json.load(f)

    # --- Validation ---
    if start_node_id not in timeline['nodes'] or end_node_id not in timeline['nodes']:
        return jsonify({"error": "Start or end node not in the specified timeline"}), 400
    if timeline['nodes'][start_node_id].get('next') != end_node_id:
        return jsonify({"error": "The selected nodes are not consecutive in this timeline"}), 400

    # --- Interpolation Logic (re-used) ---
    start_latents = torch.load(os.path.join(TENSORS_DIR, f"{start_node_id}_latent.pt"))
    start_prompts = torch.load(os.path.join(TENSORS_DIR, f"{start_node_id}_prompt_embed.pt"))
    end_latents = torch.load(os.path.join(TENSORS_DIR, f"{end_node_id}_latent.pt"))
    end_prompts = torch.load(os.path.join(TENSORS_DIR, f"{end_node_id}_prompt_embed.pt"))

    latent_interpolations = interpolate(start_latents.detach().cpu().numpy(), end_latents.detach().cpu().numpy(), steps, 'slerp')
    prompt_embed_interpolations = interpolate(start_prompts.detach().cpu().numpy(), end_prompts.detach().cpu().numpy(), steps, 'linear')

    newly_created_node_ids = []
    # We skip the first and last step as they would be duplicates of start/end nodes
    for i in range(1, steps - 1):
        alpha = i / (steps - 1.0)
        interp_latents = torch.from_numpy(latent_interpolations[i]).to(start_latents.dtype)
        interp_prompts = torch.from_numpy(prompt_embed_interpolations[i]).to(start_prompts.dtype)
        image = generate_image_from_tensors(interp_latents, interp_prompts, seed=0)
        
        new_node_id = str(uuid.uuid4())
        # Save assets...
        image_path = os.path.join(IMAGES_DIR, f"{new_node_id}.png")
        latent_path = os.path.join(TENSORS_DIR, f"{new_node_id}_latent.pt")
        prompt_embed_path = os.path.join(TENSORS_DIR, f"{new_node_id}_prompt_embed.pt")
        node_meta_path = os.path.join(NODES_DIR, f"{new_node_id}.json")
        image.save(image_path)
        torch.save(interp_latents, latent_path)
        torch.save(interp_prompts, prompt_embed_path)
        
        art_node = { "id": new_node_id, "image_path": f"/images/{new_node_id}.png", "latent_path": latent_path, "prompt_embed_path": prompt_embed_path, "model_info": { "name": MODEL_NAME, "lora": LORA_NAME, "pipeline": PIPELINE }, "prompt_text": f"In-place interpolation of {start_node_id[:4]}", "seed": None, "created_at": datetime.utcnow().isoformat(), "parent_nodes": [start_node_id, end_node_id], "interp_alpha": alpha, "rating": 'good' }
        with open(node_meta_path, 'w') as f:
            json.dump(art_node, f, indent=2)
        newly_created_node_ids.append(new_node_id)
    
    # --- Linked-List Surgery ---
    if not newly_created_node_ids:
        return jsonify({"message": "No nodes to insert (steps <= 2)."}), 200

    # Chain the new nodes together
    for i, node_id in enumerate(newly_created_node_ids):
        prev_node = newly_created_node_ids[i-1] if i > 0 else start_node_id
        next_node = newly_created_node_ids[i+1] if i < len(newly_created_node_ids) - 1 else end_node_id
        timeline['nodes'][node_id] = {"prev": prev_node, "next": next_node}
    
    # Update the original start and end nodes to point to the new chain
    timeline['nodes'][start_node_id]['next'] = newly_created_node_ids[0]
    timeline['nodes'][end_node_id]['prev'] = newly_created_node_ids[-1]

    # --- Save and Respond ---
    with open(timeline_path, 'w') as f:
        json.dump(timeline, f, indent=2)

    return jsonify(timeline)

@app.route('/v1/timelines', methods=['GET', 'POST'])
def handle_timelines():
    if request.method == 'GET':
        timelines = []
        for fname in sorted(os.listdir(TIMELINES_DIR), reverse=True):
            if fname.endswith('.json'):
                with open(os.path.join(TIMELINES_DIR, fname), 'r') as f:
                    timeline_data = json.load(f)
                    timelines.append(timeline_data)
        return jsonify(timelines)

    if request.method == 'POST': # Create a new timeline from selection
        data = request.json
        node_ids = data.get('node_ids', [])
        name = data.get('name', 'New Timeline')

        timeline_id = str(uuid.uuid4())
        timeline_path = os.path.join(TIMELINES_DIR, f"{timeline_id}.json")
        linked_nodes = {}
        start_node_id_new = None
        if node_ids:
            start_node_id_new = node_ids[0]
            for i, node_id in enumerate(node_ids):
                prev_node = node_ids[i-1] if i > 0 else None
                next_node = node_ids[i+1] if i < len(node_ids) - 1 else None
                linked_nodes[node_id] = {"prev": prev_node, "next": next_node}

        timeline = {
            "id": timeline_id,
            "name": name,
            "created_at": datetime.utcnow().isoformat(),
            "start_node_id": start_node_id_new,
            "nodes": linked_nodes
        }
        with open(timeline_path, 'w') as f:
            json.dump(timeline, f, indent=2)
        return jsonify(timeline), 201




@app.route('/v1/timelines/<timeline_id>/copy', methods=['POST'])
def copy_timeline(timeline_id):
    original_timeline_path = os.path.join(TIMELINES_DIR, f"{timeline_id}.json")
    if not os.path.exists(original_timeline_path):
        return jsonify({"error": "Original timeline not found"}), 404

    with open(original_timeline_path, 'r') as f:
        original_timeline = json.load(f)

    new_timeline_id = str(uuid.uuid4())
    new_timeline_path = os.path.join(TIMELINES_DIR, f"{new_timeline_id}.json")

    new_timeline = original_timeline.copy()
    new_timeline['id'] = new_timeline_id
    new_timeline['name'] = f"[COPY] {original_timeline.get('name', 'Unnamed Timeline')}"
    new_timeline['created_at'] = datetime.utcnow().isoformat()

    with open(new_timeline_path, 'w') as f:
        json.dump(new_timeline, f, indent=2)

    return jsonify(new_timeline), 201

@app.route('/v1/timelines/<timeline_id>', methods=['DELETE'])
def delete_timeline(timeline_id):
    timeline_path = os.path.join(TIMELINES_DIR, f"{timeline_id}.json")
    if not os.path.exists(timeline_path):
        return jsonify({"error": "Timeline not found"}), 404
    
    try:
        os.remove(timeline_path)
        return jsonify({"message": "Timeline deleted successfully"}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to delete timeline: {str(e)}"}), 500

@app.route('/v1/timelines/<timeline_id>/reverse', methods=['POST'])
def reverse_timeline(timeline_id):
    timeline_path = os.path.join(TIMELINES_DIR, f"{timeline_id}.json")
    if not os.path.exists(timeline_path):
        return jsonify({"error": "Timeline not found"}), 404

    with open(timeline_path, 'r') as f:
        timeline = json.load(f)

    # Find the current last node, which will be the new start node
    current_node_id = timeline.get('start_node_id')
    if not current_node_id:
        return jsonify(timeline) # Nothing to reverse

    last_node_id = None
    while current_node_id:
        last_node_id = current_node_id
        node_data = timeline['nodes'].get(current_node_id)
        if not node_data:
            break # Should not happen in a consistent timeline
        current_node_id = node_data.get('next')

    # Reverse the prev/next pointers for all nodes
    for node_id, links in timeline['nodes'].items():
        links['prev'], links['next'] = links.get('next'), links.get('prev')

    # Update the start_node_id to the old last node
    timeline['start_node_id'] = last_node_id

    # Save the modified timeline
    with open(timeline_path, 'w') as f:
        json.dump(timeline, f, indent=2)

    return jsonify(timeline)

if __name__ == '__main__':
    # Start the Flask app
    app.run(host='0.0.0.0', port=5001)

