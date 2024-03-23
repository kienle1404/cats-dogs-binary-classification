from torchvision import transforms
from glob import glob
from PIL import Image
import sys, os
IMAGE_WIDTH=128
IMAGE_HEIGHT=128
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)

# Write transform for image
resize_transform = transforms.Resize(size=IMAGE_SIZE)
data_dir = sys.argv[1]
out_dir = sys.argv[2]

for image_path in glob(f"{data_dir}/*/*.jpg"):
    img = Image(image_path)
    img_resize = resize_transform(img)
    