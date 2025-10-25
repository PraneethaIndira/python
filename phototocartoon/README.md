# 🎨 Pix2Pix Cartoonification Pipeline

This project implements a modular, extensible Pix2Pix-based pipeline for cartoonifying real-world images. It includes training, inference, visual checkpoints, SSIM-based evaluation, and best-model tracking — all designed for clarity, reproducibility, and creative presentation.

## 🚀 Features

- Conditional GAN architecture (Pix2Pix: U-Net Generator + PatchGAN Discriminator)
- Cartoonification of paired photo/cartoon datasets
- SSIM and perceptual loss evaluation
- Visual checkpoint saving and progress GIF generation
- Best-model tracking based on SSIM plateau
- Stylized dashboard logging to `metrics.txt`
- Modular training and inference scripts
- No external config file required — all hyperparameters are inline

## 📁 Folder Structure
