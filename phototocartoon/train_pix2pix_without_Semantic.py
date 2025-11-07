import os
import torch
from torch import nn, optim
from torchvision.utils import save_image, make_grid
import imageio.v2 as imageio
from pytorch_msssim import ssim
from models.pix2pix import Generator, Discriminator
from dataset import get_datasets
from loss.perceptual import PerceptualLoss


# ========== Config ==========
num_epochs = 100
resume_path = "checkpoints/full_state_best.pth"
use_perceptual = True
batch_size = 1
lr = 0.0002
lambda_L1 = 100
save_every = 1
revert_epoch = 60
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== Paths ==========
photo_dir = "data/train/photo"
cartoon_dir = "data/train/cartoon"
checkpoint_dir = "checkpoints"
output_dir = "outputs"
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# ========== Denormalization Helper ==========
def denormalize(tensor):
    return (tensor * 0.5) + 0.5

# ========== Dataset & Loader ==========
train_dataset, val_dataset = get_datasets(photo_dir, cartoon_dir, val_split=0.2)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ========== Models ==========
netG = Generator().to(device)
netD = Discriminator().to(device)

# ========== Losses & Optimizers ==========
criterion_GAN = nn.BCEWithLogitsLoss()
criterion_L1 = nn.L1Loss()
criterion_perceptual = PerceptualLoss(weight=0.05).to(device) if use_perceptual else None
lr = 0.0002  # or whatever value you were using originally

optimizer_G = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
# ========== Resume from Best Checkpoint ==========
start_epoch = 0
best_ssim = 0.0

if os.path.exists(resume_path):
    try:
        checkpoint = torch.load(resume_path, weights_only=False)
        netG.load_state_dict(checkpoint['netG_state_dict'])
        netD.load_state_dict(checkpoint['netD_state_dict'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        start_epoch = checkpoint['epoch']
        best_ssim = checkpoint['best_ssim']
        print(f"🔁 Resuming from checkpoint at epoch {start_epoch} with SSIM {best_ssim:.4f}")
    except RuntimeError as e:
        print(f"⚠️ Checkpoint incompatible with current architecture. Starting fresh.\n{e}")


# ========== Training Loop ==========
print("Starting training loop...")
print("🧪 Training with InstanceNorm2d in both Generator and Discriminator")
netG.train()
netD.train()
try:
    for epoch in range(start_epoch, num_epochs):
        epoch_loss_D = 0.0
        epoch_loss_G_GAN = 0.0
        epoch_loss_G_L1 = 0.0
        total_loss_G_perceptual = 0.0

        for i, (photo, cartoon) in enumerate(train_loader):
            photo = photo.to(device)
            cartoon = cartoon.to(device)

            # === Train Discriminator ===
            fake = netG(photo)
            # === Compute SSIM ===
            current_ssim = ssim(fake, cartoon, data_range=1.0).item()  # Use your SSIM function
            # === Prepare Inputs for Discriminator ===

            real_pair = torch.cat((photo, cartoon), 1)
            fake_pair = torch.cat((photo, fake.detach()), 1)
            # === Train Discriminator Conditionally ===
            update_D = current_ssim >= 0.25 or i % 2 == 0
            if update_D:
                pred_real = netD(real_pair)
                pred_fake = netD(fake_pair)

                loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real))
                loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
                loss_D = (loss_D_real + loss_D_fake) * 0.5

                optimizer_D.zero_grad()
                loss_D.backward()
                optimizer_D.step()
            else:
                loss_D = torch.tensor(0.0)  # No update this step

            # === Optional Logging ===
           # print(f"Epoch [{epoch}/{num_epochs}] | Step {i} | SSIM: {current_ssim:.4f} | D Updated: {update_D}")

            # === Train Generator ===
            fake_pair = torch.cat((photo, fake), 1)

            pred_fake = netD(fake_pair)

            loss_G_GAN = criterion_GAN(pred_fake, torch.ones_like(pred_fake))
            loss_G_L1 = criterion_L1(fake, cartoon)
            loss_G_perceptual = criterion_perceptual(fake, cartoon) if use_perceptual else 0.0
            total_loss_G_perceptual += loss_G_perceptual.item() if use_perceptual else 0.0

            loss_G = loss_G_GAN + lambda_L1 * loss_G_L1 + loss_G_perceptual

            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            # === Accumulate Losses ===
            epoch_loss_D += float(loss_D)
            #epoch_loss_D += loss_D.item()
            epoch_loss_G_GAN += loss_G_GAN.item()
            epoch_loss_G_L1 += loss_G_L1.item()

        avg_loss_G_perceptual = total_loss_G_perceptual / len(train_loader) if use_perceptual else 0.0
        print("fake shape:", fake.shape)
        print("cartoon shape:", cartoon.shape)

        # === Validation SSIM Evaluation ===
        with torch.no_grad():
            val_ssim_scores = []
            for photo, cartoon in val_loader:
                photo = photo.to(device)
                cartoon = cartoon.to(device)
                fake = netG(photo)

                score = ssim(fake, cartoon, data_range=1.0).item()
                val_ssim_scores.append(score)
            ssim_score = sum(val_ssim_scores) / len(val_ssim_scores)

        # === Log SSIM per sample ===
        with open("ssim_scores.txt", "a") as f_ssim:
            f_ssim.write(f"\nEpoch {epoch+1}:\n")
            for i, score in enumerate(val_ssim_scores):
                f_ssim.write(f"  Sample {i+1}: SSIM = {score:.4f}\n")

        # === Compute Epoch Averages ===
        avg_loss_D = epoch_loss_D / len(train_loader)
        avg_loss_G_GAN = epoch_loss_G_GAN / len(train_loader)
        avg_loss_G_L1 = epoch_loss_G_L1 / len(train_loader)

        # === Print & Log Summary ===
        summary = (
            f"Epoch [{epoch + 1}/{num_epochs}] | "
            f"D Loss: {avg_loss_D:.4f} | "
            f"G Adv: {avg_loss_G_GAN:.4f} | "
            f"G L1: {avg_loss_G_L1:.4f} | "
            f"G Perceptual: {avg_loss_G_perceptual:.4f} | "
            f"SSIM: {ssim_score:.4f}"
        )
        print(summary)

        with open("metrics.txt", "a") as f:
            f.write(summary + "\n")
            if epoch == 0:
                f.write("Epoch 1: Training started with smoother upsampling generator\n")
                if epoch == start_epoch and start_epoch > 0:
                    f.write(f"Epoch {epoch + 1}: Resumed training from checkpoint\n")

            if epoch == revert_epoch:
                f.write(f"Epoch {epoch + 1}: Reverted TTUR — lr_G = lr_D = {lr}\n")

        # === Save model and visual output every N epochs ===
        if (epoch + 1) % save_every == 0:
            torch.save(netG.state_dict(), f"{checkpoint_dir}/netG_epoch_{epoch+1}.pth")
            torch.save(netD.state_dict(), f"{checkpoint_dir}/netD_epoch_{epoch+1}.pth")

            with torch.no_grad():
                sample_idx = epoch % len(val_dataset)
                val_photo, _ = val_dataset[sample_idx]
                val_photo = val_photo.unsqueeze(0).to(device)
                val_fake = netG(val_photo)

                assert fake.shape[1] == 3, f"Generator output has {fake.shape[1]} channels, expected 3"
                assert cartoon.shape[1] == 3, f"Target image has {cartoon.shape[1]} channels, expected 3"

                print("Grid input shapes:",
                      denormalize(val_photo.cpu().squeeze(0)).shape,
                      denormalize(val_fake.cpu().squeeze(0)).shape)

                grid = make_grid([
                    denormalize(val_photo.cpu().squeeze(0)),
                    denormalize(val_fake.cpu().squeeze(0))
                ], nrow=2)

                save_image(grid, f"{output_dir}/epoch_{epoch+1}.png")

            print(f"✅ Saved models and visual output at epoch {epoch+1}")

        # === Save best model based on SSIM ===
        if ssim_score > best_ssim:
            best_ssim = ssim_score
            torch.save(netG.state_dict(), f"{checkpoint_dir}/netG_best.pth")
            torch.save({
                'epoch': epoch + 1,
                'netG_state_dict': netG.state_dict(),
                'netD_state_dict': netD.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'best_ssim': best_ssim
            }, f"{checkpoint_dir}/full_state_best.pth")
            print(f"🌟 New best model saved at epoch {epoch+1} (SSIM: {ssim_score:.4f})")

        # === Save full training state every 10 epochs ===
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'netG_state_dict': netG.state_dict(),
                'netD_state_dict': netD.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'best_ssim': best_ssim
            }, f"{checkpoint_dir}/full_state_epoch_{epoch+1}.pth")
            print(f"🗂️ Full-state checkpoint saved at epoch {epoch+1}")

except Exception as e:
    print(f"💥 Training interrupted due to error: {e}")
    print("🔁 You can restart training and it will resume from the best checkpoint.")

# ========== Generate GIF ==========
print("\nGenerating training progress GIF...")
frames = []
for epoch in range(1, num_epochs + 1):
    img_path = f"{output_dir}/epoch_{epoch}.png"
    if os.path.exists(img_path):
        frames.append(imageio.imread(img_path))

if frames:
    imageio.mimsave(f"{output_dir}/training_progress.gif", frames, duration=0.5)
    print("GIF saved to outputs/training_progress.gif")
else:
    print("No frames found — GIF not generated.")