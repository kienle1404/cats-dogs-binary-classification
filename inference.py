import sys
import torch
import cv2
from omegaconf import OmegaConf
from model import ImageClassifier
from torchvision import transforms
from PIL import Image

config = OmegaConf.load(sys.argv[1])

# load model
model = ImageClassifier(**config.model)
checkpoint_path = config.infer.checkpoint_path
model.load_state_dict(torch.load(checkpoint_path)["model"])

# data
transform = transforms.Compose([
        transforms.Resize((config.dataset.IMAGE_WIDTH, config.dataset.IMAGE_HEIGHT)),
        transforms.ToTensor()])
#[B, H, W, 3]
image_path = sys.argv[2]
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = transform(Image.fromarray(img)) / 255.0

with torch.no_grad():
    output = model(img.unsqueeze(0))
    print(output)
mapping = torch.load('experiments\\mapping.pt')
invmap = {key : value for value, key in mapping.items()}
# argmax
print(invmap[torch.argmax(output).item()])



