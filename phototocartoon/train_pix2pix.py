import os
import torch
from torch import nn, optim
from torchvision.utils import save_image, make_grid
import imageio.v2 as imageio
from pytorch_msssim import ssim
from models.pix2pix import Generator, Discriminator
from dataset import get_datasets
from loss.perceptual import PerceptualLoss
import yaml
from skimage.exposure import match_histograms
import cv2
import numpy as np
from PIL import Image
import torch.nn.functional as F
import os
import lpips


with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# ========== Config ==========
num_epochs = config["num_epochs"]
resume_path = resume_path = "checkpoints/latest.pth"#"checkpoints/full_state_best.pth"
use_perceptual = config["use_perceptual"]
use_semantic = config["use_semantic"]  # Toggle semantic mode
batch_size = 1
lr = 0.0002
lambda_L1 = 100
lambda_identity = config["lambda_identity"]
save_every = 1
revert_epoch = 60
lambda_perceptual = 0.1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
debug_mode = False  # Set to True to enable extra assertions and logging
lpips_train_model = lpips.LPIPS(net='alex').to(device)


# ========== Paths ==========
photo_dir = "data/train/photo"
cartoon_dir = "data/train/cartoon"
checkpoint_dir = "checkpoints"
output_dir = "outputs"
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
os.makedirs("debug_inputs", exist_ok=True)


# ========== Denormalization Helper ==========
def denormalize(tensor):
    return (tensor * 0.5) + 0.5
# ==================Color Transfer===============
def apply_color_transfer(fake_tensor, photo_tensor):
    fake_np = denormalize(fake_tensor.squeeze(0)).permute(1, 2, 0).cpu().numpy()
    photo_np = denormalize(photo_tensor.squeeze(0)).permute(1, 2, 0).cpu().numpy()

    # Safety checks
    assert fake_np.shape[0] > 0 and fake_np.shape[1] > 0, f"Fake image collapsed: {fake_np.shape}"
    assert photo_np.shape[0] > 0 and photo_np.shape[1] > 0, f"Photo image collapsed: {photo_np.shape}"
    assert not np.isnan(fake_np).any(), "Fake image contains NaNs"
    assert not np.isnan(photo_np).any(), "Photo image contains NaNs"

    restored = match_histograms(fake_np, photo_np, channel_axis=-1)
    return restored
# ================ Edge Enhancement ===================
def apply_edge_enhancement(image_np):
    image_uint8 = (image_np * 255).astype(np.uint8)
    edges = cv2.Canny(image_uint8, 100, 200)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    enhanced = cv2.addWeighted(image_uint8, 0.9, edges_rgb, 0.1, 0)
    return enhanced
# ===============data in batch from dataset =========================
def semantic_collate(batch):
    photos = torch.stack([b[0] for b in batch])
    cartoons = torch.stack([b[1] for b in batch])
    masks = torch.stack([b[2] for b in batch])
    patches = torch.stack([b[3] for b in batch])
    filenames = [b[4] for b in batch]
    return photos, cartoons, masks, patches, filenames

# =================mask helper ===================
def fix_mask_shape(mask):
    if mask.dim() == 3:
        mask = mask.unsqueeze(0)
    if mask.shape[1] == 1:
        mask = mask.expand(-1, 3, -1, -1)
    elif mask.shape[1] != 3:
        raise ValueError(f"Unexpected mask shape: {mask.shape}")
    return mask

def ensure_batched(t):
    return t.unsqueeze(0) if t.dim() == 3 else t


# ========== Dataset & Loader ==========
train_dataset, val_dataset = get_datasets(photo_dir, cartoon_dir, val_split=0.2, use_semantic=use_semantic)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=semantic_collate)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ========== Models ==========
netG = Generator(in_channels=8, out_channels=3, use_semantic=use_semantic).to(device)
print(netG.down1.block[0])  # Should show Conv2d(8, 64, kernel_size=4, stride=2, padding=1)
netD = Discriminator(use_semantic=use_semantic).to(device)

# ========== Losses & Optimizers ==========
criterion_GAN = nn.BCEWithLogitsLoss()
criterion_L1 = nn.L1Loss()
criterion_perceptual = PerceptualLoss(weight=0.05).to(device) if use_perceptual else None
optimizer_G = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
# ========== LPIPS Metric ==========

lpips_model = lpips.LPIPS(net='alex').to(device)
lpips_model.eval()  # LPIPS doesn't need gradients

# ========== Resume from Best Checkpoint ==========
start_epoch = 0
best_ssim = 0.0
best_lpips = float('inf')

if os.path.exists(resume_path):
    try:
        print(f"🔎 Checking for resume file at: {resume_path}")

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
torch.autograd.set_detect_anomaly(True)
#print("🧪 Training with InstanceNorm2d in both Generator and Discriminator")
netG.train()
netD.train()
try:
    for epoch in range(start_epoch, num_epochs):
        epoch_loss_D = 0.0
        epoch_loss_G_GAN = 0.0
        epoch_loss_G_L1 = 0.0
        epoch_loss_G_identity = 0.0
        total_loss_G_perceptual = 0.0

        for i, (photos, cartoons, masks, patches, filenames) in enumerate(train_loader):

            if use_semantic:
                photos = photos.to(device)
                cartoons = cartoons.to(device)
                masks = masks.to(device)
                patches = patches.to(device)
                # filenames stays as list of strings — no .to(device)
                for j in range(len(filenames)):
                   # photo = photos[j]
                   # mask = masks[j]
                   # patch = patches[j]

                    photo = ensure_batched(photos[j])
                    mask = ensure_batched(masks[j])
                    patch = ensure_batched(patches[j])

                    # Now you're safe to use photo, mask, patch

                # 🔍 Identify valid samples
                valid_indices = [
                    i for i in range(len(filenames))
                    if masks[i].sum().item() > 0 and patches[i].sum().item() > 0
                ]
                # 🛑 Skip batch if all samples are bad
                if len(valid_indices) == 0:
                    skipped = [
                        filenames[i] for i in range(len(filenames))
                        if masks[i].sum().item() == 0 or patches[i].sum().item() == 0
                    ]
                    for fname in skipped:
                        print(f"⚠️ Skipped sample: {fname}")
                    continue

                # 🧹 Filter tensors and filenames
                photos = photos[valid_indices]
                cartoons = cartoons[valid_indices]
                masks = masks[valid_indices]
                patches = patches[valid_indices]
                filenames = [filenames[i] for i in valid_indices]
                for j in range(len(filenames)):
                    photo = ensure_batched(photos[j])
                    cartoon = ensure_batched(cartoons[j])
                    mask = ensure_batched(masks[j])
                    patch = ensure_batched(patches[j])
                    patch_input = patch  # if needed separately
  # → [1, C, H, W]

                    # 🔒 Safely expand mask to 3 channels
                    if mask.shape[1] == 1:
                        mask = mask.expand(-1, 3, -1, -1)
                    elif mask.shape[1] != 3:
                        raise ValueError(f"Unexpected mask shape: {mask.shape}")

                    # 🧠 Optional: Log filename if mask or patch is empty
                    if mask.sum() == 0 or patch.sum() == 0:
                        print(f"⚠️ Skipped sample: {filenames[j]}")
                        continue


                # 🔒 Expand mask to 3 channels if needed



                # Expand mask to 3 channels
                #mask = mask.expand(-1, 3, -1, -1)  # [B, 1, H, W] → [B, 3, H, W]
                # Trim patch to 2 channels
                #patch = torch.zeros_like(photo[:, :1, :, :]).expand(-1, 2, -1, -1)  # [B, 2, H, W]
                patch = patch[:, :2, :, :]  # → [B, 2, H, W]
                # Concatenate: photo (3) + mask (3) + patch (2) = 8 channels
                assert photo.shape[1] == 3, f"Expected photo to have 3 channels, got {photo.shape[1]}"
                assert mask.shape[1] == 3, f"Expected mask to have 3 channels after expand, got {mask.shape[1]}"
                assert patch.shape[1] == 2, f"Expected patch to have 2 channels, got {patch.shape[1]}"
                input_G = torch.cat([photo, mask, patch], dim=1)
                if input_G.shape[1] != 8:
                    print(f"❌ Skipping batch — input_G has {input_G.shape[1]} channels")

                    continue
                #print("photo shape:", photo.shape)
                #print("mask shape:", mask.shape if 'mask' in locals() else "mask missing")
                #print("patch shape:", patch.shape)
                #print("input_G shape before netG:", input_G.shape)

            # 🔧 Ensure patch has 8 channels for detail_branch
                if patch.shape[1] < 8:
                    patch = F.pad(patch, (0, 0, 0, 0, 0, 8 - patch.shape[1]))  # → [B, 8, H, W]

            else:
                photos, cartoons, masks, patches, filenames = batch  # ✅ unpack

                photo = photos.to(device)
                cartoon = cartoons.to(device)
                B, C, H, W = photo.shape
                dummy_photo = torch.zeros(1, 3, H, W).to(device)  # replace H, W with actual values
                # Create dummy mask and patch to match expected input shape
                dummy_mask = torch.zeros_like(dummy_photo[:, :1, :, :]).expand(-1, 3, -1, -1)
                dummy_patch = torch.zeros_like(dummy_photo[:, :1, :, :]).expand(-1, 2, -1, -1)
                input_G = torch.cat([dummy_photo, dummy_mask, dummy_patch], dim=1)
                patch = F.pad(dummy_patch, (0, 0, 0, 0, 0, 6))  # → [B, 8, H, W]
                if input_G.shape[1] != 8:
                    print(f"❌ Skipping batch — input_G has {input_G.shape[1]} channels")
                    continue
                #patch = F.pad(dummy_patch, (0, 0, 0, 0, 0, 6))  # ✅ → [B, 8, H, W]                if patch.shape[1] < 8:
                #patch = F.pad(patch, (0, 0, 0, 0, 0, 8 - patch.shape[1]))  # → [B, 8, H, W]
            #print("🧪 input_G shape:", input_G.shape)
            #print("🧪 patch shape:", patch.shape)
            assert input_G.shape[1] == 8, f"Expected 8 channels for input_G, got {input_G.shape[1]}"
            assert patch.shape[1] == 8, f"Expected 8 channels for patch, got {patch.shape[1]}"
            try:
                fake = netG(input_G, patch)
                #print("fake shape:", fake.shape)
            except Exception as e:
                print(f"💥 netG(input_G, patch) failed: {e}")
                print(f"⚠️ Skipping batch {i} — input_G shape: {input_G.shape}, patch shape: {patch.shape}")
                continue
            # or return if outside loop
            # === Train Discriminator ===
            #print("input_G shape:", input_G.shape)  # Should be [B, 8, 256, 256]
            # === Identity Loss ===
            assert fake.shape == cartoon.shape, f"Shape mismatch: fake={fake.shape}, cartoon={cartoon.shape}"
            assert fake.shape[2] > 0 and fake.shape[3] > 0, f"Fake image collapsed: {fake.shape}"
            assert cartoon.shape[2] > 0 and cartoon.shape[3] > 0, f"Cartoon image collapsed: {cartoon.shape}"
            assert not torch.isnan(fake).any(), "Fake image contains NaNs"
            assert not torch.isnan(cartoon).any(), "Cartoon image contains NaNs"

            global_l1 = criterion_L1(fake, cartoon)
            if use_semantic:
                mask_expanded = mask.expand_as(fake)  # [B, 1, H, W] → [B, 3, H, W]
                diff = torch.abs(fake - cartoon)
                masked_l1 = torch.mean(diff * mask_expanded)

                loss_G_L1 = 0.7 * global_l1 + 0.3 * masked_l1
            else:
                loss_G_L1 = global_l1
            if use_semantic:
                cartoon_input = torch.cat([cartoon, mask, patch[:, :2, :, :]], dim=1)
                if patch.shape[1] < 8:
                    pad_channels = 8 - patch.shape[1]
                    patch = torch.cat([patch, torch.zeros(patch.size(0), pad_channels, patch.size(2), patch.size(3)).to(
                        patch.device)], dim=1)
                try:
                    identity_output = netG(cartoon_input, patch)
                except Exception as e:
                    print(f"💥 netG(cartoon_input, patch) failed: {e}")
                    identity_output = cartoon
                loss_identity = criterion_L1(identity_output, cartoon)
            else:
                loss_identity = torch.tensor(0.0, device=device)

            # === Compute SSIM ===
            #print("SSIM input shapes — fake:", fake.shape, "cartoon:", cartoon.shape)
            assert fake.shape[2] > 0 and fake.shape[3] > 0, "Fake image has zero height or width"
            assert cartoon.shape[2] > 0 and cartoon.shape[3] > 0, "Cartoon image has zero height or width"
            assert not torch.isnan(fake).any(), "Fake contains NaNs"
            assert not torch.isnan(cartoon).any(), "Cartoon contains NaNs"
            fake_norm = torch.clamp((fake + 1) / 2, 0, 1)
            cartoon_norm = torch.clamp((cartoon + 1) / 2, 0, 1)

            #print(
               # f"🧪 SSIM Debug — fake range: {fake_norm.min():.4f} to {fake_norm.max():.4f}, cartoon range: {cartoon_norm.min():.4f} to {cartoon_norm.max():.4f}")

            current_ssim = ssim(fake_norm, cartoon_norm, data_range=1.0).item()
            # === Prepare Inputs for Discriminator ===
            real_pair = torch.cat((input_G, cartoon), 1)
            #print("real_pair shape:", real_pair.shape)
            assert real_pair.shape[2] > 0 and real_pair.shape[3] > 0, "real_pair has zero height or width"
            fake_pair = torch.cat((input_G, fake.detach()), 1)
            assert real_pair.shape[1] == 11, f"Expected 11 channels for discriminator input, got {real_pair.shape[1]}"
            #assert real_pair.shape[1] == 11, f"Discriminator input mismatch: got {real_pair.shape[1]} channels"
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
                loss_D = torch.tensor(0.0)
            # === Train Generator ===

            fake_pair = torch.cat((input_G, fake), 1)
            assert fake_pair.shape[1] == 11, f"Expected 11 channels for fake_pair, got {fake_pair.shape[1]}"
            assert fake_pair.shape[2] > 0 and fake_pair.shape[3] > 0, "fake_pair has zero height or width"
            pred_fake = netD(fake_pair)
            loss_G_GAN = criterion_GAN(pred_fake, torch.ones_like(pred_fake))
            loss_G_L1 = criterion_L1(fake, cartoon)
            loss_G_perceptual = criterion_perceptual(fake, cartoon) if use_perceptual else 0.0
            if use_perceptual:
                #print("Perceptual input shapes — fake:", fake.shape, "cartoon:", cartoon.shape)
                assert fake.shape[2] >= 32 and fake.shape[3] >= 32, "Fake too small for perceptual loss"
            total_loss_G_perceptual += loss_G_perceptual.item() if use_perceptual else 0.0
            loss_G = loss_G_GAN + lambda_L1 * loss_G_L1 + lambda_perceptual * loss_G_perceptual + lambda_identity * loss_identity
            optimizer_G.zero_grad()
            #loss_G_lpips = lpips_train_model(fake, cartoon).mean()
            #loss_G += 0.1 * loss_G_lpips  # You can tune this weight
            loss_G.backward()
            optimizer_G.step()
            # === Accumulate Losses ===

            epoch_loss_D += float(loss_D)
            epoch_loss_G_GAN += loss_G_GAN.item()
            epoch_loss_G_L1 += loss_G_L1.item()
            epoch_loss_G_identity += loss_identity.item()
        input_G = None  # placeholder

        if i == 0 and epoch == start_epoch:
            print("fake shape:", fake.shape)
            print("cartoon shape:", cartoon.shape)

        # === Validation SSIM Evaluation ===

        with torch.no_grad():
            val_ssim_scores = []
            val_lpips_scores = []
            for batch in val_loader:
                if use_semantic:
                    photos, cartoons, masks, patchs, filenames = batch
                    photo = photos.to(device)
                    cartoon = cartoons.to(device)
                    mask = masks.to(device)
                    patch = patchs.to(device)

                    assert patch.dim() == 4, f"💥 patch is not 4D: {patch.shape}"

                    # Fallback if patch has fewer than 2 channels BEFORE padding
                    if patch.shape[1] < 2:
                        print(f"⚠️ Patch has insufficient channels: {patch.shape}")
                        patch_input = torch.zeros_like(photo[:, :1, :, :]).expand(-1, 2, -1, -1)
                    else:
                        if patch.shape[1] < 8:
                            patch = F.pad(patch, (0, 0, 0, 0, 0, 8 - patch.shape[1]))
                        patch_input = patch[:, :2, :, :]

                    # 🩹 Fix mask if it's single-channel
                    if mask.shape[1] == 1:
                        mask = mask.expand(-1, 3, -1, -1)

                    # 🧪 Validate shapes
                    assert photo.shape[1] == 3, f"❌ photo must have 3 channels, got {photo.shape}"
                    assert mask.shape[1] == 3, f"❌ mask must have 3 channels, got {mask.shape}"
                    assert patch_input.shape[1] == 2, f"❌ patch_input must have 2 channels, got {patch_input.shape}"

                    # Build input_G
                    input_G = torch.cat([photo, mask, patch_input], dim=1)
                    assert input_G.shape[1] == 8, f"❌ input_G has wrong channel count: {input_G.shape}"

                    if input_G.shape[1] != 8 or patch.shape[1] != 8:
                        print(f"❌ Skipping batch — input_G: {input_G.shape}, patch: {patch.shape}")
                        continue
                else:
                    photos, cartoons, masks, patches, filenames = batch
                    for j in range(len(filenames)):
                        photo = ensure_batched(photos[j].to(device))
                        cartoon = ensure_batched(cartoons[j].to(device))

                        assert photo.dim() == 4, f"photo must be 4D before slicing, got {photo.shape}"
                        dummy_mask = torch.zeros_like(photo[:, :1, :, :]).expand(-1, 3, -1, -1)
                        dummy_patch = torch.zeros_like(photo[:, :1, :, :]).expand(-1, 2, -1, -1)
                        input_G = torch.cat([photo, dummy_mask, dummy_patch], dim=1)
                        patch = torch.zeros_like(photo[:, :1, :, :]).expand(-1, 8, -1, -1)
                        #print("🧪 input_G shape:", input_G.shape)
                        #print("🧪 patch shape:", patch.shape)
                        assert input_G.dim() == 4, f"input_G must be 4D, got {input_G.shape}"
                        assert patch.dim() == 4, f"patch must be 4D, got {patch.shape}"

                        try:
                            fake = netG(input_G, patch)
                            # 🎨 Output stats
                            print(f"🎨 fake mean: {fake.mean():.4f}, std: {fake.std():.4f}")

                            # Normalize tensors to [0, 1] if needed
                            fake_norm = (fake + 1) / 2
                            cartoon_norm = (cartoon + 1) / 2

                            # Debug SSIM inputs
                            #print(f"🧪 SSIM Debug:")
                            #print(
                              #  f"fake_norm shape: {fake_norm.shape}, min: {fake_norm.min():.4f}, max: {fake_norm.max():.4f}")
                            #print(
                              #  f"cartoon_norm shape: {cartoon_norm.shape}, min: {cartoon_norm.min():.4f}, max: {cartoon_norm.max():.4f}")

                            # Compute SSIM
                            score = ssim(fake_norm, cartoon_norm, data_range=1.0).item()
                            val_ssim_scores.append(score)
                            lpips_score = lpips_model(fake, cartoon).item()
                            val_lpips_scores.append(lpips_score)
                            print(f"👁️ LPIPS score for sample {j}: {lpips_score:.4f}")
                            # ✅ Save side-by-side SSIM inputs
                            save_image(
                                torch.cat([fake_norm, cartoon_norm], dim=0),
                                f"debug_inputs/epoch_{epoch}_sample_{j}_ssim_inputs.png",
                                normalize=True
                            )

                        except Exception as e:
                            print(f"💥 Generator forward pass failed: {e}")
                            continue


            ssim_score = sum(val_ssim_scores) / (len(val_ssim_scores) + 1e-6)
            avg_lpips = sum(val_lpips_scores) / (len(val_lpips_scores) + 1e-6)
            if avg_lpips < best_lpips:
                best_lpips = avg_lpips
                torch.save(netG.state_dict(), f"{output_dir}/best_netG_lpips.pth")
                print(f"💾 Saved best LPIPS model at epoch {epoch + 1}")

            if ssim_score > best_ssim:
                best_ssim = ssim_score
                torch.save(netG.state_dict(), f"{output_dir}/best_netG_ssim.pth")
                print(f"💾 Saved best SSIM model at epoch {epoch + 1}")
        # === Log SSIM per sample ===

        with open("ssim_scores.txt", "a") as f_ssim:
            f_ssim.write(f"\nEpoch {epoch + 1}:\n")
            for i, score in enumerate(val_ssim_scores):
                f_ssim.write(f"  Sample {i + 1}: SSIM = {score:.4f}\n")
        # === Compute Epoch Averages ===

        avg_loss_D = epoch_loss_D / len(train_loader)
        avg_loss_G_GAN = epoch_loss_G_GAN / len(train_loader)
        avg_loss_G_L1 = epoch_loss_G_L1 / len(train_loader)
        avg_loss_G_identity = epoch_loss_G_identity / len(train_loader)
        avg_loss_G_perceptual = total_loss_G_perceptual / len(train_loader) if use_perceptual else 0.0
        # === Print & Log Summary ===

        summary = (
            f"Epoch [{epoch + 1}/{num_epochs}] | "
            f"D Loss: {avg_loss_D:.4f} | "
            f"G Adv: {avg_loss_G_GAN:.4f} | "
            f"G L1: {avg_loss_G_L1:.4f} | "
            f"G Identity: {avg_loss_G_identity:.4f} | "
            f"G Perceptual: {avg_loss_G_perceptual:.4f} | "
            f"SSIM: {ssim_score:.4f} | "
            f"LPIPS: {avg_lpips:.4f}"


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
                val_sample = val_dataset[sample_idx]

                # Handle semantic vs non-semantic input
                if isinstance(val_sample, tuple) and len(val_sample) == 4:
                    al_photo, val_mask, val_patch, val_filename = val_sample
                    val_photo = val_photo.unsqueeze(0).to(device)
                    val_mask = val_mask.unsqueeze(0).to(device)
                    val_patch = val_patch.unsqueeze(0).to(device)
                    val_filename = val_sample[3]  # assuming your dataset returns filename as the 4th item  # or however you're accessing it

                    # 🛑 Check for empty mask or patch BEFORE padding or slicing
                    if val_mask.sum() == 0 or val_patch.sum() == 0:
                        print(f"⚠️ Empty mask or patch — substituting dummy tensors for batch {i}")
                        with open("skipped_samples.txt", "a") as f:
                            f.write(f"{val_filename}\n")
                        val_mask = torch.zeros_like(val_photo[:, :1, :, :]).expand(-1, 3, -1, -1)
                        val_patch = torch.zeros_like(val_photo[:, :1, :, :]).expand(-1, 8, -1, -1)
                    else:
                        val_mask = val_mask.expand(-1, 3, -1, -1)
                        if val_patch.shape[1] < 8:
                            val_patch = F.pad(val_patch, (0, 0, 0, 0, 0, 8 - val_patch.shape[1]))

                    # ✅ Now safe to slice and build input_G
                    assert val_patch.dim() == 4, f"val_patch must be 4D before slicing, got {val_patch.shape}"
                    assert val_patch.shape[1] >= 2, f"val_patch must have at least 2 channels, got {val_patch.shape[1]}"
                    patch_input = val_patch[:, :2, :, :]
                    input_G = torch.cat([val_photo, val_mask, patch_input], dim=1)

                    # 🔍 Validate shapes before forward pass
                    if input_G.shape[1] != 8 or val_patch.shape[1] != 8:
                        print(f"❌ Skipping batch {i} — input_G: {input_G.shape}, patch: {val_patch.shape}")
                        torch.save(input_G.cpu(), f"debug_inputs/epoch{epoch}_batch{i}_inputG.pt")
                        torch.save(val_patch.cpu(), f"debug_inputs/epoch{epoch}_batch{i}_patch.pt")
                        continue

                    try:
                        val_fake = netG(input_G, val_patch)
                    except Exception as e:
                        print(f"💥 Generator forward pass failed (semantic): {e}")
                        continue

                else:
                    val_photo = val_sample[0].unsqueeze(0).to(device)
                    dummy_mask = torch.zeros_like(val_photo[:, :1, :, :]).expand(-1, 3, -1, -1)
                    dummy_patch = torch.zeros_like(val_photo[:, :1, :, :]).expand(-1, 8, -1, -1)
                    assert dummy_patch.dim() == 4, f"dummy_patch must be 4D before slicing, got {dummy_patch.shape}"
                    assert dummy_patch.shape[
                               1] >= 2, f"dummy_patch must have at least 2 channels, got {dummy_patch.shape[1]}"
                    patch_input = dummy_patch[:, :2, :, :]
                    input_G = torch.cat([val_photo, dummy_mask, patch_input], dim=1)

                    try:
                        val_fake = netG(input_G, dummy_patch)
                    except Exception as e:
                        print(f"💥 Generator forward pass failed (non-semantic): {e}")
                        continue  # ✅ fallback

                    #dummy_patch = F.pad(dummy_patch, (0, 0, 0, 0, 0, 6))  # → [B, 8, H, W]
                if isinstance(val_sample, tuple) and len(val_sample) == 4:
                    if isinstance(val_sample, tuple) and len(val_sample) == 4:
                        try:
                            val_fake = netG(input_G, val_patch)
                            print("val_fake shape:", val_fake.shape)
                        except Exception as e:
                            print(f"💥 Generator forward pass failed (semantic): {e}")
                            #return  # or continue if inside a loop
                else:
                    val_fake = netG(input_G, dummy_patch)  # ✅ fallback

                #val_fake = netG(input_G, dummy_patch)
                print("val_fake shape:", val_fake.shape)
                # === Post-Processing ===
                # === Post-Processing and Save ===
                if val_fake.shape[2] > 0 and val_fake.shape[3] > 0:
                    try:
                        if ssim_score < 0.1:
                            print(f"⚠️ SSIM too low — skipping enhancement")
                            enhanced_np = denormalize(val_fake.squeeze(0)).permute(1, 2, 0).cpu().numpy()
                        else:
                            restored_np = apply_color_transfer(val_fake, val_photo)
                            enhanced_np = apply_edge_enhancement(restored_np)
                        if enhanced_np.shape[0] == 1 and enhanced_np.shape[1] == 1 and enhanced_np.shape[2] == 3:
                            enhanced_np = enhanced_np.squeeze(0)

                        if enhanced_np.dtype != np.uint8:
                            enhanced_np = (enhanced_np * 255).clip(0, 255).astype(np.uint8)

                        # Save enhanced output
                        print("Saving image with shape:", enhanced_np.shape)
                        assert enhanced_np.shape[0] > 0 and enhanced_np.shape[1] > 0, "Image has zero height or width"
                        assert not np.isnan(enhanced_np).any(), "Image contains NaNs"
                        Image.fromarray(enhanced_np).save(f"{output_dir}/epoch_{epoch + 1}_enhanced.png")
                        #Image.fromarray(enhanced_np).save(f"{output_dir}/epoch_{epoch + 1}_enhanced.png")
                        print("val_photo:", val_photo.shape, "val_fake:", val_fake.shape)
                    except Exception as e:
                        print(f"⚠️ Post-processing failed: {e}")
                else:
                    print(f"⚠️ Skipping visual save — val_fake collapsed: {val_fake.shape}")

                # Save original grid for comparison
                grid = make_grid([
                    denormalize(val_photo.cpu().squeeze(0)),
                    denormalize(val_fake.cpu().squeeze(0))
                ], nrow=2)
                save_image(grid, f"{output_dir}/epoch_{epoch + 1}.png")

                if debug_mode:
                    print("Grid input shapes:", val_photo.shape, val_fake.shape)
                    assert val_fake.shape[1] == 3, "Fake output must have 3 channels"
                    save_image(grid, f"{output_dir}/epoch_{epoch + 1}_debug.png")

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
            print(f"✅ Saving full state to: {checkpoint_dir}/full_state_best.pth")
        # Always save latest checkpoint for resume
        torch.save({
            'epoch': epoch + 1,
            'netG_state_dict': netG.state_dict(),
            'netD_state_dict': netD.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'optimizer_D_state_dict': optimizer_D.state_dict(),
            'best_ssim': best_ssim
        }, f"{checkpoint_dir}/latest.pth")
        print(f"💾 Saved latest checkpoint at epoch {epoch + 1} for resume")

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
import os
from PIL import Image
import imageio

print("\nGenerating training progress GIF...")
frames = []

for epoch in range(1, num_epochs + 1):
    img_path = f"{output_dir}/epoch_{epoch}.png"
    if os.path.exists(img_path) and os.path.getsize(img_path) > 0:
        try:
            # Validate with PIL first
            with Image.open(img_path) as img:
                if img.size[0] > 0 and img.size[1] > 0:
                    frame = imageio.imread(img_path)
                    frames.append(frame)
                else:
                    print(f"⚠️ Skipping frame from epoch {epoch} — image has zero dimensions: {img.size}")
        except Exception as e:
            print(f"⚠️ Error reading frame from epoch {epoch}: {e}")
    else:
        if not os.path.exists(img_path) or os.path.getsize(img_path) == 0:
            print(f"⚠️ Skipping frame — file missing or empty: {img_path}")
            continue

if frames:
    try:
        imageio.mimsave(f"{output_dir}/training_progress.gif", frames, duration=0.5)
        print("✅ GIF saved to outputs/training_progress.gif")
    except Exception as e:
        print(f"💥 GIF generation failed: {e}")
else:
    print("⚠️ No valid frames found — GIF not generated.")

# === Export to ONNX ===
print("\nExporting generator to ONNX...")
dummy_input = torch.randn(1, 8, 256, 256).to(device)
dummy_patch = torch.randn(1, 2, 256, 256).to(device)  # or padded to 8 if needed

# If patch is padded:
dummy_patch = F.pad(dummy_patch, (0, 0, 0, 0, 0, 6))  # → [1, 8, 256, 256]

torch.onnx.export(netG, (dummy_input, dummy_patch), "generator.onnx",
                  input_names=['input', 'patch'],
                  output_names=['output'],
                  opset_version=11)