import torch
import numpy as np
import cv2
import mediapipe as mp
import torch.nn.functional as F

# Initialize Mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# Facial landmark indices for key regions
FACIAL_REGIONS = {
    "eyes": list(range(33, 133)),      # Approximate eye region
    "mouth": list(range(78, 308)),     # Approximate mouth region
    "face": list(range(0, 468))        # Full face mesh
}

def extract_mask(image_tensor, region="face"):
    """
    Extracts a binary mask for the specified facial region.
    Args:
        image_tensor: torch.Tensor of shape [3, H, W], normalized [-1, 1]
        region: one of "eyes", "mouth", "face"
    Returns:
        mask_tensor: torch.Tensor of shape [1, H, W], values in {0, 1}
    """
    # Convert to uint8 image
    image_np = ((image_tensor.permute(1, 2, 0).cpu().numpy() + 1) * 127.5).astype(np.uint8)
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Run Mediapipe
    results = face_mesh.process(image_rgb)
    h, w, _ = image_rgb.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]

        points = []
        for idx in FACIAL_REGIONS.get(region, FACIAL_REGIONS["face"]):
            lm = landmarks.landmark[idx]
            x = np.clip(int(lm.x * w), 0, w - 1)
            y = np.clip(int(lm.y * h), 0, h - 1)
            points.append([x, y])
        points = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)
    else:
        print("⚠️ No face landmarks detected — returning empty mask")

    # Convert to tensor
    mask_tensor = torch.from_numpy(mask).float().div(255.0).unsqueeze(0)  # [1, H, W]
    return mask_tensor

def extract_patch(image_tensor, mask_tensor):
    """
    Crops a patch from the image using the mask bounding box.
    Args:
        image_tensor: [3, H, W]
        mask_tensor: [1, H, W]
    Returns:
        patch_tensor: [3, H', W']
    """
    mask_np = mask_tensor.squeeze(0).cpu().numpy()
    coords = cv2.findNonZero((mask_np * 255).astype(np.uint8))
    if coords is None:
        print("⚠️ Empty mask — returning zero patch")
        return torch.zeros((3, 256, 256), dtype=image_tensor.dtype)
    x, y, w, h = cv2.boundingRect(coords)
    patch = image_tensor[:, y:y+h, x:x+w]
    # Resize patch to match input size (e.g., 256×256)
    patch = F.interpolate(patch.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False).squeeze(0)

    return patch