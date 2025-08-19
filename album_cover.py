import argparse
import random
import re

import plotly.express as px
import torch

from datetime import datetime

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from train import load_data

from PIL import Image
from PIL.PngImagePlugin import PngInfo


names = list(load_data('./ftl-fixed.csv')['ftl'])

artstyles = [
    'grainy',
    'spooky',
    'beautiful',
    'epic',
    'pixelart',
    'horror',
    'colorful',
    'gloomey',
    'haunting',
    'hellish',
    'outer space',
    'alien',
    'retro sci-fi',
    'stylized',
    'fantasy',
    'poop',
    'textured',
    'vibrant',
    'echo',
    'quiet',
    'reverberating',
    'trill',
    'meditative',
    'clear',
    'fuzz',
    'murky',
    'organized',
    'organic',
    'futurism',
    'aerospace',
    'galactic', 
]

mediums = [ 
    'painting',
    'photograph',
    'drawing',
    'watercolor',
    'digital drawing',
    'piece',
    'artwork',
    '',
]

steps = 50
pipe = StableDiffusionPipeline.from_pretrained(
    'stabilityai/stable-diffusion-2-1-base',
    torch_dtype=torch.float16
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

for i in range(len(names)):
    name = names[i]
    seed = datetime.now().timestamp()
   
    medium = random.choice(mediums)
    styles = ' '.join([medium] + [random.choice(artstyles) for _ in range(4)])
    prompt = f'{styles} piece called {name}' 

    savename = re.sub(r'[^\w.-]', '_', prompt)
    savename = f'{savename}.{str(seed)}'
    savepath = f'images/{savename}.png'
    print(savename)

    # create a generator for reproducibility; notice you don't place it on the GPU!
    generator = torch.manual_seed(seed)
    image = pipe(prompt, generator=generator, num_inference_steps=steps).images[0]
    image.save(savepath)

    image = Image.open(savepath)

    metadata = PngInfo()
    metadata.add_text("name", f"{name}")
    metadata.add_text("prompt", prompt)
    metadata.add_text("seed", str(seed))
    metadata.add_text("steps", str(steps))
    image.save(savepath, pnginfo=metadata)
    image = Image.open(savepath)
    print(image.text)
