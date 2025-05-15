import os
import glob
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import kornia
import kornia.filters as kf
import kornia.losses as kl

class EdgeAwareLoss(nn.Module):  
    def __init__(self, edge_weight=0.5):
        super(EdgeAwareLoss, self).__init__()
        self.edge_weight = edge_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.edge_loss = kl.TotalVariation()
        
    def forward(self, pred, edge_pred, target):
        # Main segmentation loss (BCE + Dice)
        seg_loss = 0.3 * self.bce(pred, target) + 0.7 * self.dice(pred, target)
        
        # Edge loss using total variation
        edge_loss = self.edge_loss(edge_pred)
          # Calculate edge map from target for supervision
        target = target.float()  # Ensure target is float
        target_edges = torch.sqrt(
            kf.sobel(target, normalized=True)[0].pow(2) +
            kf.sobel(target, normalized=True)[1].pow(2)
        )
        edge_supervision_loss = self.bce(edge_pred, target_edges)
        
        # Combine all losses
        total_loss = seg_loss + self.edge_weight * (edge_loss + edge_supervision_loss)
        return total_loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        # Flatten predictions and targets
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice

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
class ClothSegmentationDataset(Dataset):    
    def __init__(self, img_dir, mask_dir, img_size, img_names, augment=True):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.img_names = img_names
        self.augment = augment
        
        # Edge-aware augmentations for better border detection
        self.albumentations_transform = A.Compose([
            A.Resize(img_size[0], img_size[1]),            A.OneOf([
                A.Affine(scale=1.15, rotate=10, translate_percent=0.1, p=0.7),
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.3)
            ], p=0.5),
            A.OneOf([
                A.GaussNoise(p=0.3),
                A.MultiplicativeNoise(p=0.3),
            ], p=0.2),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=0.1),
            ], p=0.2),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.RandomBrightnessContrast(),            
            ], p=0.3),
            A.HorizontalFlip(p=0.5),
            ToTensorV2(),
        ], additional_targets={'mask': 'mask'})
        
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize(img_size, interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])

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


# --- MODEL (Edge-Aware Segmentation Model) ---
class SegmentationModel(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(SegmentationModel, self).__init__()
        
        # Encoder with edge-aware features
        self.enc1 = EdgeAwareEncoder(n_channels, 64)
        self.enc2 = EdgeAwareEncoder(64, 128)
        self.enc3 = EdgeAwareEncoder(128, 256)
        self.enc4 = EdgeAwareEncoder(256, 512)
        
        # Bridge
        self.bridge = nn.Sequential(
            EdgeAwareConv(512, 1024),
            EdgeAwareConv(1024, 512)
        )
        
        # Decoder with skip connections and edge refinement
        self.dec4 = EdgeAwareDecoder(1024, 256)
        self.dec3 = EdgeAwareDecoder(512, 128)
        self.dec2 = EdgeAwareDecoder(256, 64)
        self.dec1 = EdgeAwareDecoder(128, 32)
        
        # Edge detection branch
        self.edge_branch = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1)
        )
        
        # Final segmentation head
        self.final = nn.Sequential(
            nn.Conv2d(33, 32, 3, padding=1),  # 33 channels: 32 from decoder + 1 from edge branch
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, n_classes, 1)
        )
        
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))
        
        # Bridge
        bridge = self.bridge(F.max_pool2d(enc4, 2))
        
        # Decoder with skip connections
        dec4 = self.dec4(bridge, enc4)
        dec3 = self.dec3(dec4, enc3)
        dec2 = self.dec2(dec3, enc2)
        dec1 = self.dec1(dec2, enc1)
        
        # Edge detection
        edge_out = self.edge_branch(dec1)
        
        # Combine features with edge information
        combined = torch.cat([dec1, edge_out], dim=1)
        
        # Final segmentation
        out = self.final(combined)
        
        return out, edge_out

class EdgeAwareEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EdgeAwareEncoder, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.edge_attention = EdgeAttentionModule(out_channels)
        
    def forward(self, x):
        x = self.double_conv(x)
        x = self.edge_attention(x)
        return x

class EdgeAwareDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EdgeAwareDecoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, 2, stride=2)
        self.conv = EdgeAwareEncoder(in_channels, out_channels)
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class EdgeAttentionModule(nn.Module):
    def __init__(self, channels):
        super(EdgeAttentionModule, self).__init__()
        self.edge_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        edge_attention = self.edge_conv(x)
        return x * edge_attention

class EdgeAwareConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EdgeAwareConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.edge_attention = EdgeAttentionModule(out_channels)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.edge_attention(x)
        return x


# --- TRAINING ---
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    for images, masks in tqdm(dataloader, desc="Training"):
        images = images.to(device).float()
        masks = masks.to(device).float()
        outputs, edge_outputs = model(images)
        loss = criterion(outputs, edge_outputs, masks)
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
            outputs, edge_outputs = model(images)
            loss = criterion(outputs, edge_outputs, masks)
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
    val_img_names = [all_img_names[i] for i in val_indices]    # Create Datasets and DataLoaders with memory optimizations
    train_dataset = ClothSegmentationDataset(IMG_DIR, MASK_DIR, IMG_SIZE, img_names=train_img_names, augment=True)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                                num_workers=2, pin_memory=True)

    val_dataset = ClothSegmentationDataset(IMG_DIR, MASK_DIR, IMG_SIZE, img_names=val_img_names, augment=False)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                              num_workers=2, pin_memory=True)

    test_dataset = ClothSegmentationDataset(TEST_IMG_DIR, TEST_MASK_DIR, IMG_SIZE,
                                             img_names=sorted([f for f in os.listdir(TEST_IMG_DIR) if
                                                               os.path.isfile(os.path.join(TEST_IMG_DIR, f))]),
                                             augment=False)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)    # Model, Loss, Optimizer
    model = SegmentationModel().to(DEVICE)
    criterion = EdgeAwareLoss(edge_weight=0.5)  # Edge-aware loss
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