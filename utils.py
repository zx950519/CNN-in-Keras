from PIL import Image

import os

def resize(input_path, output_path, width, height, type):
    img = Image.open(input_path)
    out = img.resize((width, height), Image.ANTIALIAS)
    out.save(output_path, type)