import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os
import time
from collections import deque
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_socketio import SocketIO
import base64
import json

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Keep track of active sessions
active_sessions = {}

# Define preprocessing steps for shirt images
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 192)),  # Common aspect ratio for fashion images (4:3)
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),  # Add robustness to lighting
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, dropout_prob=0.1):
        super(UNet, self).__init__()
        self.dropout_prob = dropout_prob

        def CBR(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
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

def segment_shirt(img_np):
    try:
        # Convert to RGB for preprocessing
        img_pil = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        img_tensor = preprocess(img_pil).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(img_tensor)
            mask_prob = output.squeeze().cpu().numpy()

        # Apply threshold with higher confidence
        binary_mask = (mask_prob > 0.3).astype(np.uint8) * 255
        
        if binary_mask.sum() == 0:
            print("Warning: Model produced empty segmentation mask")
            return np.zeros((img_np.shape[0], img_np.shape[1]), dtype=np.uint8)
        
        # Resize the mask to original image size
        mask_resized = cv2.resize(binary_mask, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        # Post-processing to clean up the mask
        kernel = np.ones((3,3), np.uint8)
        mask_resized = cv2.morphologyEx(mask_resized, cv2.MORPH_CLOSE, kernel)  # Fill small holes
        mask_resized = cv2.morphologyEx(mask_resized, cv2.MORPH_OPEN, kernel)   # Remove small noise
        
        # Find contours and keep only the largest one (the shirt)
        contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            mask_resized = np.zeros_like(mask_resized)
            cv2.drawContours(mask_resized, [largest_contour], -1, 255, -1)
        
        return mask_resized
    except Exception as e:
        print(f"Error during segmentation: {e}")
        return np.zeros((img_np.shape[0], img_np.shape[1]), dtype=np.uint8)

class VirtualTryOn:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.fps_history = deque(maxlen=150)
        self.cap = None
        self.shirt_image = None
        self.shirt_mask = None
        self.shirt_no_bg = None
        self.shirt_name = None
        
    def start_camera(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            return self.cap.isOpened()
        return True
    
    def stop_camera(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def process_frame(self):
        if not self.cap or not self.shirt_no_bg is not None:
            return None

        ret, frame = self.cap.read()
        if not ret:
            return None

        # Process frame (copied from original logic)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            # ... (rest of the processing logic remains the same)
            # Apply virtual try-on
            processed_frame = self.apply_virtual_tryon(frame, landmarks)
            return processed_frame

        return frame

    def apply_virtual_tryon(self, frame, landmarks):
        # ... (copy all the virtual try-on logic from the original file)
        # This includes the shirt positioning, resizing, and blending
        return frame

    def set_garment(self, image_path):
        self.shirt_image = cv2.imread(image_path)
        if self.shirt_image is not None:
            self.shirt_mask = segment_shirt(self.shirt_image)
            self.shirt_no_bg = cv2.bitwise_and(self.shirt_image, self.shirt_image, mask=self.shirt_mask)
            self.shirt_name = os.path.basename(image_path)
            return True
        return False

# Initialize the UNet model
model = UNet(in_channels=3, out_channels=1, dropout_prob=0.1)
# Load model weights
weights_path = "new_segmentation_unet.pth"
try:
    checkpoint = torch.load(weights_path, map_location=DEVICE)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(DEVICE)
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

@app.route('/api/start-session', methods=['POST'])
def start_session():
    session_id = str(time.time())
    active_sessions[session_id] = VirtualTryOn()
    if active_sessions[session_id].start_camera():
        return jsonify({"session_id": session_id})
    return jsonify({"error": "Failed to start camera"}), 500

@app.route('/api/set-garment', methods=['POST'])
def set_garment():
    session_id = request.form.get('session_id')
    if session_id not in active_sessions:
        return jsonify({"error": "Invalid session"}), 400
    
    image_path = request.form.get('image_path')
    if not image_path:
        return jsonify({"error": "No image path provided"}), 400
    
    if active_sessions[session_id].set_garment(image_path):
        return jsonify({"success": True})
    return jsonify({"error": "Failed to load garment"}), 500

@app.route('/api/save-screenshot', methods=['POST'])
def save_screenshot():
    session_id = request.form.get('session_id')
    if session_id not in active_sessions:
        return jsonify({"error": "Invalid session"}), 400
    
    session = active_sessions[session_id]
    frame = session.process_frame()
    if frame is not None:
        screenshots_dir = "screenshots"
        if not os.path.exists(screenshots_dir):
            os.makedirs(screenshots_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(screenshots_dir, f"virtual_tryon_{timestamp}.png")
        cv2.imwrite(filename, frame)
        return jsonify({"filename": filename})
    return jsonify({"error": "Failed to save screenshot"}), 500

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('start_stream')
def handle_start_stream(data):
    session_id = data.get('session_id')
    if session_id not in active_sessions:
        return
    
    session = active_sessions[session_id]
    while True:
        frame = session.process_frame()
        if frame is not None:
            # Convert frame to base64 string
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            socketio.emit('frame', {'frame': frame_base64}, room=request.sid)
        socketio.sleep(0.033)  # ~30 FPS

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')
    # Cleanup session if needed

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
