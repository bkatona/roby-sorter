import math
import sys
from PIL import Image
import pathlib
from resizeimage import resizeimage
max_size = (255,)

folder_dir = "C:/Users/brian/Documents/Roby/pruned_dataset/metal"
for input_img_path in pathlib.Path(folder_dir).iterdir():
    with Image.open(input_img_path) as image:
        size_list = image.size + max_size
        min_size = min(size_list)
        size = [min_size, min_size]
        cover = resizeimage.resize_cover(image, size)
        #greyscale = cover.convert('L')
        #greyscale.save(input_img_path, image.format)
        cover.save(input_img_path, image.format)
        
