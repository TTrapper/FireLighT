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
#pipe.enable_model_cpu_offload()
pipe.safety_checker = None
print("Model loaded.")

# --- Prompt Generation Data (from artserver.py) ---
names = [
'flyby teacher lizzard',
'first tongue leaves',
'foresaw technopia learning',
'for titan lords',
'freeze the lotus',
'fright them lunars',
'faithful tidal leaf',
'frost taking lead',
'folly told lamb',
'faithful throne lab',
'forest toward lens',
'ferret taking lab',
'forbidden tower leans',
'folly throne larch',
'fusing the learned',
'fuse thought legend',
'frigid transhuman learns',
'for throne lustrous',
'faded teach lizzard',
'flora tricks lethal',
'for thunderous lurking',
'first teleport lair',
'final try lion',
'foresaw tomb landmark',
'fog teleportation lunacy',
'falcon took life',
'froth take lurking',
'fern tells lapel',
'ficus tropic landing',
'folly teaches learning',
'fertilize them leaders',
'from tornado locomotion',
'flaming tractor lilted',
'flame transhuman luminance',
'fuzz thunders loyalty',
'fourth time lapel',
'felt tundra lion',
'ferret teacher lucid',
'footsteps to lilac',
'flowering tundra light',
'forge trance lucid',
'fading trance limber',
'faintest teacher lilted',
'fermion throne lens',
'feral tiger lurker',
'fleet tongues lilted',
'farout toward lion',
'flowers thunder lively',
'fellowship terror larch',
'ferns thought large',
'foresaw terrible languish',
'flounder temple lore',
'faithful too lacking',
'full timeless luck',
'flesh tales late',
'feel teaching lemur',
'fleet tiger law',
'frog tundra listens',
'fellowship tries lemur',
'footsteps take lion',
'frost temple lupine',
'frog tropical listener',
'fear try late',
'foul trick learning',
'flora torpedoes land',
'flowering toad limitation',
'frozen tombstone length',
'fireball tie lurk',
'flower to learn',
'fuzz thorn luster',
'free timeless luna',
'fly terrifying lake',
'fear their leopard',
'fellowship taunting lies',
'fermi tendrils luminent',
'foxglove twilight lander',
'ficus took listener',
'frigate trauma levitation',
'fruit terrestrial lake',
'fermion trick limits',
'find tidal lullaby',
'failing tulips lament',
'frenzied trauma luck',
'frigid trembling luck',
'fail terrain lost',
'for terrifying love',
'fox that lonely',
'flowers toward luminance',
'faithful tombs low',
'final thought locomotion',
'forbidden toxic lament',
'fang taunting lexicon',
'finding tides landed',
'fight the life',
'fading thought learns',
'fatal thick lamb',
'forbidden thrill lucifer',
'failing tricks lucifer',
'farout the lichen',
'fourth terrestrial love',
'frictionless terror lure',
'fail tongue lives',
'fungi took leave',
'frictionless trepidation lissome',
'final terror landed',
'from tethys low',
'from taken lepers',
'feint the luminance',
'fright torch leaps',
'freak tries living',
'fertilized tomb lemongrass',
'fir torch luck',
'feline thought lends',
'fooled tethys limits',
'fleeing teleportation lull',
'fox taking love',
'flyby thriller lush',
'formation thriller leaped',
'Feynman\'s talisman limbers',
'foresaw tiger longevity',
'fatally toward lair',
'fang temples lilt',
'fungi teach learned',
'flamingo tendril luxury',
'fossil terraformed loft',
'forceps teach living',
'faded transcendent logic',
'foxglove traumatized learner',
'father toadstool luster',
'flee thorn lissome',
'fusion twilight lizzard',
'fractal thorn lake',
'ferret tongues lead',
'forge tried limelight',
'fuzz titans live',
'feral thy learn',
'fruit taking love',
'fast tricks luring',
'from tortured lichen',
'fungi trance lumin',
'frenzied tungsten lilt',
'fern thunder late',
'feint trepidation lord',
'frost taunting lexicon',
'fuel teaches liers',
'fungi tries liquor',
'field terrified lilacs',
'foxglove talisman lead',
'fungi thunders lachrymose',
'fermented talisman learns',
'from terrified languish',
'fast taken leaps',
'foresee them lullabies',
'faithful tropic lilies',
'fornax tektite leaf',
'frost tropic lullaby',
'ferret tease learn',
'forged terra love',
'full tension learned',
'fermions teaching lemur',
'flora taro lacking',
'footsteps teasing lushness',
'falling torture lucid',
'field tesseract loquacious',
'forming their learning',
'fight twilight lullaby',
'fortold terraform lunatics',
'flyby tides lavender',
'forge terra lilies',
'fungi thunders lost',
'fly terrible lofts',
'force talisman lustrous',
'falcon tries levitation',
'free the lost',
'fall tracking lighthouse',
'forest transhumans listen',
'fermentation tractor leopard',
'fungus temples learning',
'frictionless tombstones lilted',
'fade toward lucid',
'flight trepidation low',
'flashlight tendrils liquid',
'fail the larch',
'fuel terrain lingers',
'faith thrilling limber',
'frenzied the liberty',
'faintest tungsten lamb',
'folly thrilled lenses',
'formation teased listener',
'foresee temperance lachrymose',
'fermions try luck',
'fighter toad luminent',
'freedom transcends limbs',
'forge thrilling lotus',
'foxglove twilight lurker',
'forged talisman lavender',
'fight trick love',
'foresight takes luck',
'froth terrain luring',
'failing tabor lens',
'fall through loop',
'fly titan loyal',
'fossil tech lost',
'fear taken limits',
'felt toad luminescence',
'fighting terra lemur',
'flyby taught lumi',
'falcon tropics local',
'faith transfigures loyal',
'frightening try\'s legacy',
'farout the lead',
'fuse taking lepers',
'follow thistle\'s lever',
'fly tithing lead',
'fuel talisman landing',
'fractal truth lemon',
'frozen tongues lament',
'footsteps teas lamb',
'foul take limit',
'formation to learn',
'future toad leapt',
'forbidden transcendent lushness',
'fighting timeless limits',
'frigid time lava',
'fog town lab',
'flee tired loyalty',
'fell teasel lilted',
'fourth tulsi liberation',
'foul terra loving',
'frothy trembling lies',
'forget time legacy',
'freedom teleportation loon',
'fell tithes laser',
'forging tension liberty',
'falling tetrarch lore',
'forging trickster lyrics',
'full tide lamb',
'frightening tornado landed',
'fermi terraforms lament',
'forgive taro locomotion',
'feel torpedoes lucid',
'frost terrain leap',
'foreboding too languorous',
'foresaw temperance lithosphere',
'far toad luminance',
'fuel throne lively',
'freeze tides laudable',
'fermi\'s tethys lens',
'flowers  try  lithosphere',
'frothy tropic lotus',
'fossilize the laser',
'faithful tropic legacy',
'flight tropic leisure',
'father took long',
'force tithes lacking',
'fossil thick labyrinth',
'faintest technopia lexicon',
'frog titans loyalty',
'frenzied triton lemongrass',
'fell tortured loyalty',
'fallen thunder lairs',
'father tendrils laughs',
'faith teaching leisure',
'fractal tidal lavender',
'far temple liberty',
'forged thunders lying',
'flowering to lure',
'flaming terror love',
'full toad limit',
'foreboding turkey laugh',
'flesh tries leisure',
'forbidden tomb lamb',
'forged terror lurks',
'feared tiger lunges',
'freaks torment luminosity',
'field terrifyingly lost',
'field through listen',
'fuzz timeless loving',
'fox tropical lake',
'fermi traumas lilt',
'foxglove tempered lilt',
'freaks tricks learn',
'flesh toward lier',
'forgiveness transcends lull',
'fuzzy too loud',
'feel tombs lurk',
'frenzied thunderous laments',
'forgiving techno lair',
'feral terrible lizzard',
'free tesseract landed',
'fright twilight lair',
'flying temple lighthouses',
'fear thrilling languish',
'forgive the lemurs',
'frothing time loops',
'foresight time loops',
'free two lovers',
'forgiven thrills leave',
'far tropic longing',
]

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
    



def generate_n_images(count):
    """Generates a given number of new art nodes in a thread."""
    print(f"Starting background generation of {count} images.")
    for i in range(count):
        try:
            print(f"Generating image {i+1}/{count}...")
            generate_new_art_node()
        except Exception as e:
            print(f"Error during batch generation: {e}")
    print(f"Finished generating {count} images.")

@app.route('/v1/generator/generate', methods=['POST'])
def trigger_generation():
    count = request.json.get('count', 1)
    if not isinstance(count, int) or count <= 0 or count > 100:
         return jsonify({"error": "Invalid 'count'. Must be an integer between 1 and 100."}), 400

    # Run generation in a background thread to not block the request
    thread = threading.Thread(target=generate_n_images, args=(count,))
    thread.daemon = True
    thread.start()

    return jsonify({"message": f"Started generating {count} images in the background."}), 202


# --- API Endpoints ---

@app.route('/')
def index():
    """Serves the main UI."""
    return render_template('art_engine_ui.html')

@app.route('/images/<path:filename>')
# ... rest of your routes
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

@app.route('/v1/interpolate', methods=['POST'])
def interpolate_nodes():
    data = request.json
    start_node_id = data.get('start_node_id')
    end_node_id = data.get('end_node_id')
    steps = int(data.get('steps', 30))

    # Load start and end node tensors
    start_latents = torch.load(os.path.join(TENSORS_DIR, f"{start_node_id}_latent.pt"))
    start_prompts = torch.load(os.path.join(TENSORS_DIR, f"{start_node_id}_prompt_embed.pt"))
    end_latents = torch.load(os.path.join(TENSORS_DIR, f"{end_node_id}_latent.pt"))
    end_prompts = torch.load(os.path.join(TENSORS_DIR, f"{end_node_id}_prompt_embed.pt"))

    # Interpolate latents and prompt embeds using utils.interpolate
    latent_interpolations = interpolate(start_latents.detach().cpu().numpy(), end_latents.detach().cpu().numpy(), steps, 'slerp')
    prompt_embed_interpolations = interpolate(start_prompts.detach().cpu().numpy(), end_prompts.detach().cpu().numpy(), steps, 'linear')

    newly_created_node_ids = []

    for i in range(steps):
        alpha = i / (steps - 1)
        
        # Get interpolated tensors
        interp_latents = torch.from_numpy(latent_interpolations[i]).to(start_latents.dtype)
        interp_prompts = torch.from_numpy(prompt_embed_interpolations[i]).to(start_prompts.dtype)

        # Create new node
        image = generate_image_from_tensors(interp_latents, interp_prompts, seed=0) # Using a fixed seed for now, can be made dynamic
        
        new_node_id = str(uuid.uuid4())
        image_path = os.path.join(IMAGES_DIR, f"{new_node_id}.png")
        latent_path = os.path.join(TENSORS_DIR, f"{new_node_id}_latent.pt")
        prompt_embed_path = os.path.join(TENSORS_DIR, f"{new_node_id}_prompt_embed.pt")
        node_meta_path = os.path.join(NODES_DIR, f"{new_node_id}.json")

        image.save(image_path)
        torch.save(interp_latents, latent_path)
        torch.save(interp_prompts, prompt_embed_path)
        
        art_node = {
            "id": new_node_id,
            "image_path": f"/images/{new_node_id}.png",
            "latent_path": latent_path,
            "prompt_embed_path": prompt_embed_path,
            "model_info": {
                "name": MODEL_NAME,
                "lora": LORA_NAME,
                "pipeline": PIPELINE
            },
            "prompt_text": None,
            "seed": None,
            "created_at": datetime.utcnow().isoformat(),
            "parent_nodes": [start_node_id, end_node_id],
            "interp_alpha": alpha,
            "rating": 'good', # Auto-rate interpolated as good
        }
        with open(node_meta_path, 'w') as f:
            json.dump(art_node, f, indent=2)

        newly_created_node_ids.append(new_node_id)
    
    # Create a new timeline for this interpolation
    timeline_id = str(uuid.uuid4())
    timeline_path = os.path.join(TIMELINES_DIR, f"{timeline_id}.json")

    # Build the new linked-list structure for nodes
    linked_nodes = {}
    start_node_id_new = None
    if newly_created_node_ids:
        start_node_id_new = newly_created_node_ids[0]
        for i, node_id in enumerate(newly_created_node_ids):
            prev_node = newly_created_node_ids[i-1] if i > 0 else None
            next_node = newly_created_node_ids[i+1] if i < len(newly_created_node_ids) - 1 else None
            linked_nodes[node_id] = {"prev": prev_node, "next": next_node}

    timeline = {
        "id": timeline_id,
        "name": f"Interpolation {start_node_id[:4]} to {end_node_id[:4]}",
        "created_at": datetime.utcnow().isoformat(),
        "start_node_id": start_node_id_new,
        "nodes": linked_nodes
    }
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

if __name__ == '__main__':
    # Start the Flask app
    app.run(host='0.0.0.0', port=5001)
