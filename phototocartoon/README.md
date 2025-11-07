## 📘 `README`

# 🎨 Enhanced Pix2Pix Cartoonification Pipeline


This project performs stylized cartoonification of real-world images using a modified Pix2Pix framework. It supports multi-channel inputs including RGB, edge maps, masks, and semantic maps, and includes robust preprocessing for mobile and WhatsApp images.It includes training, inference, visual checkpoints, SSIM-based evaluation, and best-model tracking — all designed for clarity, reproducibility, and creative presentation.

## 🧠 Features
```markdown
- ✅ 8-channel input: RGB + Edge + Mask + Semantic
- ✅ U-Net Generator with skip connections
- ✅ PatchGAN Discriminator
- ✅ JPEG compression simulation for mobile robustness
- ✅ Edge-aware sharpening and denoising
- ✅ Visual debugging of input channels
- ✅ Modular preprocessing pipeline
- ✅ Cartoonification of paired photo/cartoon datasets
- ✅ SSIM and perceptual loss evaluation
- ✅ Visual checkpoint saving and progress GIF generation
- ✅ Best-model tracking based on SSIM plateau
- ✅ Stylized dashboard logging to `metrics.txt`
- ✅ Modular training and inference scripts
- ✅ No external config file required — all hyperparameters are inline

```

## 🗂️ File Structure

```
phototocartoon/
├── train_pix2pix.py         # Training script
├── predict.py               # Inference script
├── dataset.py               # Paired image loader
├── models/
│   └── pix2pix.py           # Generator & Discriminator
├── utils/
│   └── transforms.py        # Preprocessing & augmentation
├── checkpoints/
│   └── best.pth             # Trained model weights
├── input/                   # Inference images
├── output/                  # Stylized results
└── debug/                   # Saved input channels
```

---

## 🚀 Inference

```bash
python predict.py
```

Make sure your input images are placed in the `input/` folder. Outputs will be saved to `output/`.

---

## 🧪 Training

```bash
python train_pix2pix.py
```

Training uses paired photo-cartoon datasets. You can customize augmentations in `dataset.py`.

---

## 🛠️ Preprocessing Highlights

- JPEG compression:
  ```python
  encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 30]
  result, encimg = cv2.imencode('.jpg', img_np, encode_param)
  img_np = cv2.imdecode(encimg, 1)
  ```
- Edge-aware sharpening:
  ```python
  img_np = cv2.GaussianBlur(img_np, (3, 3), 0)
  img_np = cv2.addWeighted(img_np, 1.5, cv2.GaussianBlur(img_np, (0, 0), 3), -0.5, 0)
  ```

---

## 📦 Requirements

See `requirements.txt` 

---

## 📸 Sample Output

Stylized cartoon images with enhanced structure and color blending. Robust across internet and mobile inputs.

---

## 🧑‍💻 Author

Built and refined by [Praneetha](https://github.com/PraneethaIndira/python) with a focus on robustness, modularity, and visual clarity.



