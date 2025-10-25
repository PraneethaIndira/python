import matplotlib.pyplot as plt

# ========== Config ==========
log_file = "metrics.txt"
save_path = "metrics_plot.png"

# ========== Parse Log ==========
epochs = []
d_loss = []
g_adv = []
g_l1 = []
g_perceptual = []
ssim_scores = []

with open(log_file, "r") as f:
    for line in f:
        if "Epoch [" in line:
            parts = line.strip().split("|")
            epoch = int(parts[0].split("[")[1].split("/")[0])
            d = float(parts[1].split(":")[1])
            g_adv_loss = float(parts[2].split(":")[1])
            g_l1_loss = float(parts[3].split(":")[1])
            g_perc = float(parts[4].split(":")[1])
            ssim = float(parts[5].split(":")[1])

            epochs.append(epoch)
            d_loss.append(d)
            g_adv.append(g_adv_loss)
            g_l1.append(g_l1_loss)
            g_perceptual.append(g_perc)
            ssim_scores.append(ssim)

# ========== Plot ==========
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(epochs, d_loss, label="D Loss", color="red")
plt.plot(epochs, g_adv, label="G Adv Loss", color="blue")
plt.plot(epochs, g_l1, label="G L1 Loss", color="green")
plt.plot(epochs, g_perceptual, label="G Perceptual", color="purple")
plt.title("Loss Trends")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(epochs, ssim_scores, label="SSIM", color="black")
plt.title("SSIM Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("SSIM")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(save_path)
print(f"📈 Metrics plot saved to {save_path}")