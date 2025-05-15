import os
import glob
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add after imports
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, outputs, targets):
        outputs = outputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (outputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (outputs.sum() + targets.sum() + self.smooth)
        return 1 - dice

class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        
    def forward(self, outputs, targets):
        bce_loss = self.bce(outputs, targets)
        dice_loss = self.dice(outputs, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss

# Add early stopping helper class
class EarlyStopping:
    def __init__(self, patience=7, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# --- CONFIG --- "C:\Users\HP\Desktop\Others\new\DATA\train\cloth"
IMG_DIR = r'C:\Users\HP\Desktop\Others\new\DATA\train\cloth'
MASK_DIR = r'C:\Users\HP\Desktop\Others\new\DATA\train\cloth-mask'
TEST_IMG_DIR = r'C:\Users\HP\Desktop\Others\new\DATA\test\cloth'
TEST_MASK_DIR = r'C:\Users\HP\Desktop\Others\new\DATA\test\cloth-mask'
BATCH_SIZE = 8  # Reduced for 4GB GPU
NUM_EPOCHS = 50  # Reduced to prevent memory issues
LEARNING_RATE = 1e-4  # Standard learning rate
PATIENCE = 7  # Reduced patience for faster convergence
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_SAVE_PATH = 'new_segmentation_unet.pth'
IMG_SIZE = (256, 192)
WEIGHT_DECAY = 1e-5
DROPOUT_PROB = 0.1
VAL_SPLIT = 0.2  # Fraction of training data to use for validation

# --- DATASET ---
class ClothSegmentationDataset(Dataset):    def __init__(self, img_dir, mask_dir, img_size, img_names, augment=True):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.img_names = img_names
        self.augment = augment
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize(img_size, interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])
        # Simplified augmentations for memory efficiency
        self.albumentations_transform = A.Compose([            A.Resize(img_size[0], img_size[1]),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.ShiftScaleRotate(scale_limit=0.15, rotate_limit=10, shift_limit=0.1, p=0.5),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.3),
            ToTensorV2(),
        ], additional_targets={'mask': 'mask'})

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.augment:
            augmented = self.albumentations_transform(image=np.array(image), mask=np.array(mask))
            image = augmented['image'].float()
            mask = augmented['mask'].unsqueeze(0).float()
        else:
            image = self.transform(image).float()
            mask = self.mask_transform(mask).float()

        mask = (mask > 0.5).float()
        return image, mask


# --- MODEL (U-Net) ---
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, dropout_prob=0.1):
        super(UNet, self).__init__()
        self.dropout_prob = dropout_prob        def CBR(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        self.enc1 = CBR(in_channels, 64)
        self.enc2 = CBR(64, 128)
        self.enc3 = CBR(128, 256)
        self.enc4 = CBR(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.middle = nn.Sequential(
            CBR(512, 1024),
            nn.Dropout(dropout_prob)
        )
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = CBR(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = CBR(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = CBR(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = CBR(128, 64)
        self.final = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        m = self.middle(self.pool(e4))
        d4 = self.up4(m)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        out = self.final(d1)
        return torch.sigmoid(out)


# --- TRAINING ---
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    for images, masks in tqdm(dataloader, desc="Training"):
        images = images.to(device).float()
        masks = masks.to(device).float()
        outputs = model(images)
        loss = criterion(outputs, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)


# --- VALIDATION ---
def validate(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validation"):
            images = images.to(device).float()
            masks = masks.to(device).float()
            outputs = model(images)
            loss = criterion(outputs, masks)
            epoch_loss += loss.item()
    return epoch_loss / len(dataloader)


def main():
    print("Starting training with improved configuration...")
    print(f"Using device: {DEVICE}")
    print(f"Number of epochs: {NUM_EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Early stopping patience: {PATIENCE}")

    # Get all image names
    all_img_names = sorted([f for f in os.listdir(IMG_DIR) if os.path.isfile(os.path.join(IMG_DIR, f))])
    num_total = len(all_img_names)
    indices = np.arange(num_total)
    np.random.shuffle(indices)

    # Split indices for training and validation
    val_size = int(VAL_SPLIT * num_total)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    train_img_names = [all_img_names[i] for i in train_indices]
    val_img_names = [all_img_names[i] for i in val_indices]

    # Create Datasets and DataLoaders
    train_dataset = ClothSegmentationDataset(IMG_DIR, MASK_DIR, IMG_SIZE, img_names=train_img_names, augment=True)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    val_dataset = ClothSegmentationDataset(IMG_DIR, MASK_DIR, IMG_SIZE, img_names=val_img_names, augment=False)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    test_dataset = ClothSegmentationDataset(TEST_IMG_DIR, TEST_MASK_DIR, IMG_SIZE,
                                             img_names=sorted([f for f in os.listdir(TEST_IMG_DIR) if
                                                               os.path.isfile(os.path.join(TEST_IMG_DIR, f))]),
                                             augment=False)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)    # Model, Loss, Optimizer
    model = UNet(dropout_prob=DROPOUT_PROB).to(DEVICE)
    criterion = CombinedLoss(bce_weight=0.3, dice_weight=0.7)  # More weight on Dice loss for better boundaries
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    print(f"Training on: {DEVICE}")

    best_val_loss = float('inf')
    early_stopping = EarlyStopping(patience=PATIENCE, min_delta=1e-4)
    for epoch in range(NUM_EPOCHS):
        train_loss = train(model, train_dataloader, criterion, optimizer, DEVICE)
        val_loss = validate(model, val_dataloader, criterion, DEVICE)

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        # Save the model if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Validation loss improved, saving model to {MODEL_SAVE_PATH}")

        # Check for early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    print(f"Best Validation Loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()