import torch
from torchvision.utils import save_image
from PIL import Image, ImageDraw
import torchvision.transforms as T
from models.pix2pix import Generator
import os
import cv2
import numpy as np

# Paths
input_dir = "data/inference/photo"
output_dir = "data/inference/cartoonified"
checkpoint_path = "checkpoints/netG_best.pth"

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(output_dir, exist_ok=True)

# Load model
netG = Generator(8, 3, use_semantic=True).to(device)
netG.load_state_dict(torch.load(checkpoint_path))
netG.eval()

# Transforms
rgb_transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

mask_transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor()
])

def load_edge(img):
    img_np = np.array(img)
    edge = cv2.Laplacian(img_np, cv2.CV_64F)
    edge = np.uint8(np.clip(edge, 0, 255))  # Convert back to uint8
    edge = cv2.cvtColor(edge, cv2.COLOR_BGR2GRAY)  # Ensure single channel
    edge = Image.fromarray(edge).convert("L")
    return mask_transform(edge)

def load_mask(img):
    # Dummy mask: white circle in center (replace with actual mask logic)
    mask = Image.new("L", img.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((img.size[0]//4, img.size[1]//4, 3*img.size[0]//4, 3*img.size[1]//4), fill=255)
    return mask_transform(mask)

def preprocess_mobile_image(img):
    img = img.convert("RGB")
    img = img.resize((256, 256), Image.LANCZOS)
    img_np = np.array(img)

    noise_level = np.std(img_np)
    # ✅ Simulate WhatsApp-style JPEG compression
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 30]
    result, encimg = cv2.imencode('.jpg', img_np, encode_param)
    img_np = cv2.imdecode(encimg, 1)

    # ✅ Add edge-aware sharpening here
    img_np = cv2.GaussianBlur(img_np, (3, 3), 0)
    img_np = cv2.addWeighted(img_np, 1.5, cv2.GaussianBlur(img_np, (0, 0), 3), -0.5, 0)

    # Optional: Denoising for WhatsApp artifacts

    if noise_level < 15:
        img_np = cv2.bilateralFilter(img_np, d=9, sigmaColor=100, sigmaSpace=100)
    else:
        img_np = cv2.bilateralFilter(img_np, d=7, sigmaColor=75, sigmaSpace=75)

    return Image.fromarray(img_np)

def load_semantic(img):
    semantic = img.convert("RGB")  # Convert grayscale to RGB
    return rgb_transform(semantic)  # Now returns 3-channel tensor

# Inference
with torch.no_grad():
    for fname in os.listdir(input_dir):
        path = os.path.join(input_dir, fname)
        img = Image.open(path).convert("RGB")
        img = preprocess_mobile_image(img)

        rgb = rgb_transform(img)
        edge = load_edge(img)
        mask = load_mask(img)
        semantic_tensor = load_semantic(img)
        # ✅ Save debug images
        debug_dir = "debug"
        os.makedirs(debug_dir, exist_ok=True)
        save_image(rgb, os.path.join(debug_dir, f"{fname}_rgb.png"))
        save_image(edge, os.path.join(debug_dir, f"{fname}_edge.png"))
        save_image(mask, os.path.join(debug_dir, f"{fname}_mask.png"))
        save_image(semantic_tensor, os.path.join(debug_dir, f"{fname}_semantic.png"))

        input_tensor = torch.cat([rgb, edge, mask, semantic_tensor], dim=0).unsqueeze(0).to(device)  # [1, 8, H, W]
        output = netG(input_tensor)
        save_image(output * 0.5 + 0.5, os.path.join(output_dir, fname))
        print(f"✅ Cartoonified: {fname}")