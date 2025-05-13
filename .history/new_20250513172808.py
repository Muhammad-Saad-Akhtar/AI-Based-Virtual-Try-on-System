import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import filedialog
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# --- Define your UNet model architecture here 
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

# Initialize your UNet model with dropout
model = UNet(in_channels=3, out_channels=1, dropout_prob=0.1)

# Load the saved weights from local path
weights_path = "new_segmentation_unet.pth"
try:
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.eval()
    print(f"Successfully loaded weights from: {weights_path}")
except FileNotFoundError:
    print(f"Error: Weights file not found at: {weights_path}")
    print("Make sure the weights file is in the same directory as this script")
    exit()
except Exception as e:
    print(f"Error loading weights: {e}")
    exit()

# Define the preprocessing steps for the shirt image
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((320, 320)),  # Use the size your model was trained on (IMG_SIZE from training.py)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Initialize MediaPipe Pose and Camera ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# --- Function to open file dialog and select an image ---
def select_image():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select Shirt Image", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    return file_path

# --- Get the shirt image using file dialog ---
shirt_image_path = select_image()

if not shirt_image_path:
    print("No image selected. Exiting.")
    exit()

# --- Load shirt image ---
shirt_image = cv2.imread(shirt_image_path, cv2.IMREAD_UNCHANGED)
if shirt_image is None:
    print("Error loading image. Please check the file path.")
    exit()

# --- Replace the extract_shirt function with UNet inference ---
def segment_shirt(img_np):
    img_pil = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    img_tensor = preprocess(img_pil).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        mask_prob = output.squeeze().cpu().numpy()

    # Threshold the probability map to get a binary mask
    binary_mask = (mask_prob > 0.5).astype(np.uint8) * 255

    # Resize the mask to the original image size
    mask_resized = cv2.resize(binary_mask, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_LINEAR)

    return mask_resized

# Get the segmentation mask from the UNet model
shirt_mask = segment_shirt(shirt_image)

# Apply the mask to extract the shirt (assuming white background)
shirt_no_bg = cv2.bitwise_and(shirt_image, shirt_image, mask=shirt_mask)

# Show the extracted shirt and its mask (for debugging)
cv2.imshow("Extracted Shirt (UNet)", shirt_no_bg)
cv2.imshow("Shirt Mask (UNet)", shirt_mask)

# --- Set up webcam and window ---
cap = cv2.VideoCapture(0)
cv2.namedWindow('Virtual Try-On', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Virtual Try-On', 800, 600)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = results.pose_landmarks.landmark
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

        shoulder_distance_horizontal = abs(left_shoulder.x - right_shoulder.x)
        frame_width = frame.shape[1]
        shirt_width = int(shoulder_distance_horizontal * frame_width * 1.6)
        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_y = (left_hip.y + right_hip.y) / 2
        torso_height = abs(hip_y - shoulder_y)
        frame_height = frame.shape[0]
        shirt_height = int(shirt_width * (shirt_no_bg.shape[0] / shirt_no_bg.shape[1]) * 1.5)

        shirt_resized = cv2.resize(shirt_no_bg, (shirt_width, shirt_height))
        mask_resized = cv2.resize(shirt_mask, (shirt_width, shirt_height))

        shoulder_midpoint_y_pixel = int((left_shoulder.y + right_shoulder.y) / 2 * frame_height)
        shirt_y = int(shoulder_midpoint_y_pixel + 0.3 * frame_height - shirt_height / 2)
        shirt_x = int((left_shoulder.x + right_shoulder.x) / 2 * frame_width - shirt_width / 2)
        shirt_x = max(0, min(shirt_x, frame_width - shirt_width))
        shirt_y = max(0, min(shirt_y, frame_height - shirt_height))
        shirt_end_x = shirt_x + shirt_resized.shape[1]
        shirt_end_y = shirt_y + shirt_resized.shape[0]
        if shirt_end_x > frame_width:
            shirt_end_x = frame_width
            shirt_width = shirt_end_x - shirt_x
            shirt_resized = cv2.resize(shirt_no_bg, (shirt_width, shirt_height))
            mask_resized = cv2.resize(shirt_mask, (shirt_width, shirt_height))
        if shirt_end_y > frame_height:
            shirt_end_y = frame_height
            shirt_height = shirt_end_y - shirt_y
            shirt_resized = cv2.resize(shirt_no_bg, (shirt_width, shirt_height))
            mask_resized = cv2.resize(shirt_mask, (shirt_width, shirt_height))

        try:
            shirt_region = frame[shirt_y:shirt_end_y, shirt_x:shirt_end_x]
            resized_mask = cv2.resize(mask_resized, (shirt_region.shape[1], shirt_region.shape[0])).astype(np.uint8)
            resized_shirt = cv2.resize(shirt_resized, (shirt_region.shape[1], shirt_region.shape[0]))

            # Ensure the mask has the same number of channels as the shirt region (grayscale to BGR)
            if len(resized_mask.shape) == 2:
                resized_mask = cv2.cvtColor(resized_mask, cv2.COLOR_GRAY2BGR)

            # Use the mask to blend the shirt with the background
            alpha = resized_mask / 255.0
            frame_roi = frame[shirt_y:shirt_end_y, shirt_x:shirt_end_x].astype(float)
            shirt_overlay = (resized_shirt.astype(float) * alpha + frame_roi * (1 - alpha)).astype(np.uint8)
            frame[shirt_y:shirt_end_y, shirt_x:shirt_end_x] = shirt_overlay

        except ValueError:
            print("Error: Shirt region out of bounds (still).")

    cv2.imshow('Virtual Try-On', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()