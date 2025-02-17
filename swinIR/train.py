import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from SwinIR.models.network_swinir import SwinIR

# =======================
#  CONFIGURATION
# =======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
EPOCHS = 200
LEARNING_RATE = 1e-4
IMAGE_SIZE = 128  # Training patch size
SCALE_FACTOR = 4  # Super-resolution factor

# Paths
DATASET_PATH = "dataset"
TRAIN_LR_PATH = os.path.join(DATASET_PATH, "train_LR")
TRAIN_HR_PATH = os.path.join(DATASET_PATH, "train_HR")
VAL_LR_PATH = os.path.join(DATASET_PATH, "val_LR")
VAL_HR_PATH = os.path.join(DATASET_PATH, "val_HR")
MODEL_SAVE_PATH = "experiments/swinir_cctv.pth"

# =======================
#  DATASET LOADER (Fixed)
# =======================

class SuperResolutionDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, transform=None, image_size=(256, 256)):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.transform = transform
        self.image_size = image_size

        self.lr_images = sorted([f for f in os.listdir(self.lr_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.hr_images = sorted([f for f in os.listdir(self.hr_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

        # Ensure equal number of LR and HR images
        if len(self.lr_images) != len(self.hr_images):
            print(f"⚠️ Mismatch: {len(self.lr_images)} LR images, {len(self.hr_images)} HR images")
            min_size = min(len(self.lr_images), len(self.hr_images))
            self.lr_images = self.lr_images[:min_size]
            self.hr_images = self.hr_images[:min_size]

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        lr_image_path = os.path.join(self.lr_dir, self.lr_images[idx])
        hr_image_path = os.path.join(self.hr_dir, self.hr_images[idx])

        lr_image = cv2.imread(lr_image_path, cv2.IMREAD_COLOR)
        hr_image = cv2.imread(hr_image_path, cv2.IMREAD_COLOR)

        if lr_image is None or hr_image is None:
            print(f"⚠️ Skipping unreadable file: {lr_image_path} or {hr_image_path}")
            return self.__getitem__((idx + 1) % len(self.hr_images))

        # Convert images to RGB and resize
        lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)
        hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)

        lr_image = cv2.resize(lr_image, self.image_size)  # Resize images to fixed size
        hr_image = cv2.resize(hr_image, self.image_size)

        lr_image = torch.tensor(lr_image).permute(2, 0, 1).float() / 255.0  # Normalize and reshape
        hr_image = torch.tensor(hr_image).permute(2, 0, 1).float() / 255.0

        return lr_image, hr_image

# =======================
#  DATA TRANSFORMATIONS
# =======================
transform = transforms.Compose([
    transforms.ToTensor(),   # Convert to PyTorch tensor
])

train_dataset = SuperResolutionDataset(TRAIN_LR_PATH, TRAIN_HR_PATH, transform)
val_dataset = SuperResolutionDataset(VAL_LR_PATH, VAL_HR_PATH, transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

print(f"✅ Training Set: {len(train_dataset)} images")
print(f"✅ Validation Set: {len(val_dataset)} images")

# =======================
#  MODEL SETUP
# =======================
model = SwinIR(upscale=SCALE_FACTOR, in_chans=3, img_size=IMAGE_SIZE, window_size=8, img_range=1.0)
model.to(DEVICE)

criterion = nn.L1Loss()  # L1 Loss for super-resolution
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))

# =======================
#  TRAINING FUNCTION
# =======================
def train(model, train_loader, val_loader, epochs):
    best_psnr = 0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        with tqdm(train_loader, unit="batch") as tepoch:
            for lr_imgs, hr_imgs in tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}/{epochs}")

                lr_imgs, hr_imgs = lr_imgs.to(DEVICE), hr_imgs.to(DEVICE)

                optimizer.zero_grad()
                sr_imgs = model(lr_imgs)

                loss = criterion(sr_imgs, hr_imgs)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(train_loader)
        print(f"🔵 Epoch [{epoch+1}/{epochs}] | Avg Loss: {avg_loss:.6f}")

        # Validate & Save Best Model
        val_psnr = validate(model, val_loader)
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"✅ Best model saved with PSNR: {best_psnr:.2f}")

# =======================
#  VALIDATION FUNCTION
# =======================
def validate(model, val_loader):
    model.eval()
    total_psnr = 0
    total_ssim = 0
    with torch.no_grad():
        for lr_imgs, hr_imgs in val_loader:
            lr_imgs, hr_imgs = lr_imgs.to(DEVICE), hr_imgs.to(DEVICE)
            sr_imgs = model(lr_imgs)

            # Convert tensors to numpy for PSNR & SSIM calculation
            hr_imgs_np = hr_imgs.cpu().numpy().squeeze().transpose(1, 2, 0)
            sr_imgs_np = sr_imgs.cpu().numpy().squeeze().transpose(1, 2, 0)

            # Fix data type for SSIM
            hr_imgs_np = hr_imgs_np.astype(np.float64)
            sr_imgs_np = sr_imgs_np.astype(np.float64)

            # Compute PSNR & SSIM
            psnr_value = psnr(hr_imgs_np, sr_imgs_np, data_range=1.0)
            ssim_value = ssim(hr_imgs_np, sr_imgs_np, multichannel=True)

            total_psnr += psnr_value
            total_ssim += ssim_value

    avg_psnr = total_psnr / len(val_loader)
    avg_ssim = total_ssim / len(val_loader)
    print(f"📊 Validation - PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f}")
    return avg_psnr

# =======================
#  TRAIN THE MODEL
# =======================
if __name__ == "__main__":
    print("🚀 Training SwinIR for CCTV Image Enhancement...")
    train(model, train_loader, val_loader, EPOCHS)
    print("✅ Training Complete. Best model saved at:", MODEL_SAVE_PATH)