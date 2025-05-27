# preprocessing.py
import numpy as np
from PIL import Image
from rembg import remove

def remove_bg_and_white(img):
    if isinstance(img, str):
        img = Image.open(img)
    no_bg_img = remove(img)
    white_bg = Image.new("RGBA", no_bg_img.size, (255, 255, 255, 255))
    white_bg.paste(no_bg_img, (0, 0), no_bg_img)
    return white_bg.convert("RGB")

def preprocess_image(img: Image.Image, target_size=(100, 100)) -> np.ndarray:
    img = remove_bg_and_white(img)

    img = img.convert("RGB")        # Convert to RGB explicitly (important if image has alpha or palette)
    img = img.resize(target_size)   # Resize with PIL
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.ascontiguousarray(img_array.copy())  # Copy to own data & contiguous
    return np.expand_dims(img_array, axis=0)  # Add batch dimension