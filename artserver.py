from flask import Flask, render_template, request, send_file
import os
import random
from PIL import Image
import base64
import shutil

app = Flask(__name__)


def random_image():
    img_dir = "./images"
    img_list = os.listdir(img_dir)
    img_path = os.path.join(img_dir, random.choice(img_list))
    return img_path

@app.post('/move_to_good_folder')
def move_to_copy_folder():
    img_filepath = request.form['filepath']
    img_filename = os.path.basename(img_filepath)
    bad_images_dir = "./good_images"
    if os.path.exists(img_filepath):
        if not os.path.exists(bad_images_dir):
            os.makedirs(bad_images_dir)
        # Move the image to the bad_images folder
        shutil.move(img_filepath, os.path.join(bad_images_dir, img_filename)) 
        return {'message': 'Image copied to good folder'}
    else:
        return {'message': 'Image not found'}

@app.post('/move_to_bad_folder')
def move_to_bad_folder():
    img_filepath = request.form['filepath']
    img_filename = os.path.basename(img_filepath)
    bad_images_dir = "./bad_images"
    if os.path.exists(img_filepath):
        if not os.path.exists(bad_images_dir):
            os.makedirs(bad_images_dir)
        # Move the image to the bad_images folder
        shutil.move(img_filepath, os.path.join(bad_images_dir, img_filename)) 
        return {'message': 'Image moved to bad_images folder'}
    else:
        return {'message': 'Image not found'}

@app.post('/move_to_meh_folder')
def move_to_meh_folder():
    img_filepath = request.form['filepath']
    img_filename = os.path.basename(img_filepath)
    meh_images_dir = "./meh_images"
    if os.path.exists(img_filepath):
        if not os.path.exists(meh_images_dir):
            os.makedirs(meh_images_dir)
        # Move the image to the meh_images folder
        shutil.move(img_filepath, os.path.join(meh_images_dir, img_filename)) 
        return {'message': 'Image moved to meh_images folder'}
    else:
        return {'message': 'Image not found'}

@app.get('/next_image')
def next_image():
    img_path = random_image()
    return random_image()

@app.get('/get_image')
def get_image():
    img_path = request.args['filepath']
    with open(img_path, 'rb') as img_file:
        img_data = img_file.read()
    img = Image.open(img_path)
    img_metadata = img.info
    name = 'FTL' if 'name' not in img_metadata else img_metadata['name']
    img.close()
    return {'name':name, 'image':base64.b64encode(img_data).decode('utf8')}

@app.route("/")
def hello_world():
    return render_template('art.html')

