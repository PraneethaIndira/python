import torch
from torchvision.utils import save_image
from PIL import Image
import torchvision.transforms as T
from models.pix2pix import Generator
import os

# Paths
input_dir = "data/inference/photo"
output_dir = "data/inference/cartoonified"
checkpoint_path = "checkpoints/netG_best.pth"

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(output_dir, exist_ok=True)

# Load model
netG = Generator(3, 3).to(device)
netG.load_state_dict(torch.load(checkpoint_path))
netG.eval()

# Transform
transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,))
])

# Inference
with torch.no_grad():
    for fname in os.listdir(input_dir):
        path = os.path.join(input_dir, fname)
        img = Image.open(path).convert("RGB")
        input_tensor = transform(img).unsqueeze(0).to(device)
        output = netG(input_tensor)
        save_image(output * 0.5 + 0.5, os.path.join(output_dir, fname))
        print(f"✅ Cartoonified: {fname}")