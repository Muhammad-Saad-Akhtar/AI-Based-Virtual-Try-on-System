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

# --- CONFIG ---
IMG_DIR = r'C:\Users\HP\Desktop\Others\new\DATA\train\cloth'
MASK_DIR = r'C:\Users\HP\Desktop\Others\new\DATA\train\cloth-mask'
TEST_IMG_DIR = r'C:\Users\HP\Desktop\Others\new\DATA\test\cloth'
TEST_MASK_DIR = r'C:\Users\HP\Desktop\Others\new\DATA\test\cloth-mask'
BATCH_SIZE = 8
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_SAVE_PATH = 'new_segmentation_unet.pth'
IMG_SIZE = (320, 320)
WEIGHT_DECAY = 1e-5
DROPOUT_PROB = 0.1
VAL_SPLIT = 0.2

# --- DATASET ---
class ClothSegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size, img_names, augment=True):
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
        
        self.albumentations_transform = A.Compose([
            A.Resize(img_size[0], img_size[1]),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                A.RandomGamma(gamma_limit=(70, 130), p=0.5),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=0.5)
            ], p=0.5),
            A.OneOf([
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.5),
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
            ], p=0.5),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            ], p=0.3),
            ToTensorV2(),
        ], additional_targets={'mask': 'mask'})

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')
        except Exception as e:
            print(f"Error loading image {img_name}: {str(e)}")
            return torch.zeros(3, *self.img_size), torch.zeros(1, *self.img_size)

        if self.augment:
            image_np = np.array(image)
            mask_np = np.array(mask)
            augmented = self.albumentations_transform(image=image_np, mask=mask_np)
            image = augmented['image'].float() / 255.0
            mask = augmented['mask'].unsqueeze(0).float() / 255.0
        else:
            image = self.transform(image)
            mask = self.mask_transform(mask)

        mask = (mask > 0.2).float()
        
        return image, mask


# --- MODEL (U-Net with Attention) ---
class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        attention_mask = self.attention(x)
        return x * attention_mask

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, dropout_prob=0.1):
        super().__init__()
        
        def CBR(in_ch, out_ch, kernel_size=3, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size, padding=padding),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout_prob)
            )

        self.enc1 = CBR(in_channels, 64)
        self.enc2 = CBR(64, 128)
        self.enc3 = CBR(128, 256)
        self.enc4 = CBR(256, 512)
        
        self.att1 = AttentionBlock(64)
        self.att2 = AttentionBlock(128)
        self.att3 = AttentionBlock(256)
        self.att4 = AttentionBlock(512)
        
        self.pool = nn.MaxPool2d(2)
        
        self.bridge = nn.Sequential(
            CBR(512, 1024),
            AttentionBlock(1024),
            nn.Dropout2d(dropout_prob)
        )
        
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = CBR(1024, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = CBR(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = CBR(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = CBR(128, 64)
        
        self.final = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        e1 = self.enc1(x)
        e1_att = self.att1(e1)
        e2 = self.enc2(self.pool(e1))
        e2_att = self.att2(e2)
        e3 = self.enc3(self.pool(e2))
        e3_att = self.att3(e3)
        e4 = self.enc4(self.pool(e3))
        e4_att = self.att4(e4)
        
        bridge = self.bridge(self.pool(e4))
        
        d4 = self.up4(bridge)
        d4 = torch.cat([d4, e4_att], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3_att], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2_att], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1_att], dim=1)
        d1 = self.dec1(d1)
        
        return self.final(d1)


# --- LOSS FUNCTION ---
class BoundaryLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
    def forward(self, pred, target):
        if pred.is_cuda:
            self.sobel_x = self.sobel_x.cuda()
            self.sobel_y = self.sobel_y.cuda()
        
        # Calculate edges
        pred_edges_x = F.conv2d(pred, self.sobel_x, padding=1)
        pred_edges_y = F.conv2d(pred, self.sobel_y, padding=1)
        target_edges_x = F.conv2d(target, self.sobel_x, padding=1)
        target_edges_y = F.conv2d(target, self.sobel_y, padding=1)
        
        pred_edges = torch.sqrt(pred_edges_x**2 + pred_edges_y**2)
        target_edges = torch.sqrt(target_edges_x**2 + target_edges_y**2)
        
        # MSE loss on the edges
        return F.mse_loss(pred_edges, target_edges)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        pred = pred.contiguous()
        target = target.contiguous()
        
        intersection = (pred * target).sum(dim=2).sum(dim=2)
        loss = (1 - ((2. * intersection + self.smooth) /
                     (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + self.smooth)))
        return loss.mean()


# --- METRICS ---
def calculate_metrics(pred, target):
    pred = (pred > 0.5).float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + 1e-7) / (union + 1e-7)
    
    dice = (2. * intersection + 1e-7) / (pred.sum() + target.sum() + 1e-7)
    
    correct = (pred == target).float().sum()
    total = torch.numel(pred)
    accuracy = correct / total
    
    return iou.item(), dice.item(), accuracy.item()

def test(model, dataloader, device):
    model.eval()
    total_iou = 0
    total_dice = 0
    total_accuracy = 0
    
    print("\nTesting the model...")
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Testing"):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            iou, dice, accuracy = calculate_metrics(outputs, masks)
            
            total_iou += iou
            total_dice += dice
            total_accuracy += accuracy
    
    num_batches = len(dataloader)
    avg_iou = total_iou / num_batches
    avg_dice = total_dice / num_batches
    avg_accuracy = total_accuracy / num_batches
    
    print(f"\nTest Results:")
    print(f"IoU Score: {avg_iou:.4f}")
    print(f"Dice Coefficient: {avg_dice:.4f}")
    print(f"Pixel Accuracy: {avg_accuracy:.4f}")
    
    return avg_iou, avg_dice, avg_accuracy


# --- TRAINING ---
def train(model, dataloader, criterion, dice_criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    for images, masks in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        bce_loss = criterion(outputs, masks)
        dice_loss = dice_criterion(outputs, masks)
        loss = 0.5 * bce_loss + 0.5 * dice_loss
        
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
    all_img_names = sorted([f for f in os.listdir(IMG_DIR) if os.path.isfile(os.path.join(IMG_DIR, f))])
    num_total = len(all_img_names)
    indices = np.arange(num_total)
    np.random.shuffle(indices)

    val_size = int(VAL_SPLIT * num_total)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    train_img_names = [all_img_names[i] for i in train_indices]
    val_img_names = [all_img_names[i] for i in val_indices]

    train_dataset = ClothSegmentationDataset(IMG_DIR, MASK_DIR, IMG_SIZE, img_names=train_img_names, augment=True)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    val_dataset = ClothSegmentationDataset(IMG_DIR, MASK_DIR, IMG_SIZE, img_names=val_img_names, augment=False)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    test_dataset = ClothSegmentationDataset(TEST_IMG_DIR, TEST_MASK_DIR, IMG_SIZE,
                                          img_names=sorted([f for f in os.listdir(TEST_IMG_DIR) if
                                                            os.path.isfile(os.path.join(TEST_IMG_DIR, f))]),
                                          augment=False)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = UNet(dropout_prob=DROPOUT_PROB).to(DEVICE)
    criterion = nn.BCELoss()
    dice_criterion = DiceLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    print(f"Training on: {DEVICE}")
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        train_loss = train(model, train_dataloader, criterion, dice_criterion, optimizer, DEVICE)
        val_loss = validate(model, val_dataloader, criterion, DEVICE)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        if val_loss < best_val_loss:
            improvement = (best_val_loss - val_loss) / best_val_loss * 100
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
            }, MODEL_SAVE_PATH)
            print(f"Validation loss improved by {improvement:.2f}%, saving model to {MODEL_SAVE_PATH}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience: {patience_counter}/{patience}")
        
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break
    
    print(f"\nTraining completed. Best Validation Loss: {best_val_loss:.4f}")
    
    print("\nLoading best model for testing...")
    best_model = UNet(dropout_prob=DROPOUT_PROB).to(DEVICE)
    checkpoint = torch.load(MODEL_SAVE_PATH)
    best_model.load_state_dict(checkpoint['model_state_dict'])
    
    test_iou, test_dice, test_accuracy = test(best_model, test_dataloader, DEVICE)
    
    stats = {
        'best_val_loss': best_val_loss,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'final_lr': optimizer.param_groups[0]['lr'],
        'test_metrics': {
            'iou': test_iou,
            'dice': test_dice,
            'accuracy': test_accuracy
        }
    }
    torch.save(stats, 'training_stats.pth')


if __name__ == "__main__":
    main()