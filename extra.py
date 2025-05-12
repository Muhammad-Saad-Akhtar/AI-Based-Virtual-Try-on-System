import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import filedialog

# Initialize MediaPipe Pose and Camera
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Function to open file dialog and select an image
def select_image():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(title="Select Shirt Image", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    return file_path

# Get the shirt image using file dialog
shirt_image_path = select_image()

if not shirt_image_path:
    print("No image selected. Exiting.")
    exit()

# Load shirt image
shirt_image = cv2.imread(shirt_image_path, cv2.IMREAD_UNCHANGED)
if shirt_image is None:
    print("Error loading image. Please check the file path.")
    exit()

# Define a function to extract the shirt from the image with white background
def extract_shirt(shirt_image):
    # Convert the image to grayscale to detect the white background
    gray = cv2.cvtColor(shirt_image, cv2.COLOR_BGR2GRAY)

    # Apply a binary threshold to create a mask where the white areas are black (background)
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)  # Threshold value of 240 for white background

    # Perform morphological transformations to clean the mask (remove small noise)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Close small gaps
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # Remove small noise

    # Use the mask to extract the shirt from the image
    shirt_no_bg = cv2.bitwise_and(shirt_image, shirt_image, mask=mask)
    return shirt_no_bg, mask

shirt_no_bg, shirt_mask = extract_shirt(shirt_image)

# Show the extracted shirt and its mask (for debugging purposes)
cv2.imshow("Extracted Shirt", shirt_no_bg)
cv2.imshow("Shirt Mask", shirt_mask)

# Set up webcam
cap = cv2.VideoCapture(0)

# Create a resizable window
cv2.namedWindow('Virtual Try-On', cv2.WINDOW_NORMAL)

# Define the desired window size (adjust these values as needed)
window_width = 800
window_height = 600

# Resize the window
cv2.resizeWindow('Virtual Try-On', window_width, window_height)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Convert frame to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    # Draw pose landmarks on the frame
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Extract body landmarks
        landmarks = results.pose_landmarks.landmark

        # Get the coordinates of key body landmarks (e.g., shoulders, hips)
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

        # Calculate the horizontal distance between shoulders
        shoulder_distance_horizontal = abs(left_shoulder.x - right_shoulder.x)
        frame_width = frame.shape[1]

        # Estimate the shirt width (adjusted scaling factor)
        shirt_width = int(shoulder_distance_horizontal * frame_width * 1.6)

        # Calculate the vertical distance between shoulders and hips
        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_y = (left_hip.y + right_hip.y) / 2
        torso_height = abs(hip_y - shoulder_y)
        frame_height = frame.shape[0]

        # Estimate the shirt height (adjusted scaling factor)
        shirt_height = int(shirt_width * (shirt_no_bg.shape[0] / shirt_no_bg.shape[1]) * 1.5)

        # Resize the shirt and mask together to ensure they align
        shirt_resized = cv2.resize(shirt_no_bg, (shirt_width, shirt_height))
        mask_resized = cv2.resize(shirt_mask, (shirt_width, shirt_height))

        # Adjust the vertical position to move the shirt down (with a fixed offset)
        shoulder_midpoint_y_pixel = int((left_shoulder.y + right_shoulder.y) / 2 * frame_height)
        shirt_y = int(shoulder_midpoint_y_pixel + 0.3 * frame_height - shirt_height / 2)

        # Ensure the shirt stays within the frame boundaries
        shirt_x = int((left_shoulder.x + right_shoulder.x) / 2 * frame_width - shirt_width / 2)
        shirt_x = max(0, min(shirt_x, frame_width - shirt_width))
        shirt_y = max(0, min(shirt_y, frame_height - shirt_height))

        # Calculate the end points of the shirt region
        shirt_end_x = shirt_x + shirt_resized.shape[1]
        shirt_end_y = shirt_y + shirt_resized.shape[0]

        # Ensure the end points are also within the frame
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

        # Now extract the shirt region and overlay, ensuring mask dimensions match
        try:
            shirt_region = frame[shirt_y:shirt_end_y, shirt_x:shirt_end_x]
            resized_mask_inv = cv2.resize(mask_resized, (shirt_region.shape[1], shirt_region.shape[0])).astype(np.uint8)
            resized_shirt = cv2.resize(shirt_resized, (shirt_region.shape[1], shirt_region.shape[0]))
            resized_mask = cv2.resize(shirt_mask, (shirt_region.shape[1], shirt_region.shape[0])).astype(np.uint8)


            shirt_bg = cv2.bitwise_and(shirt_region, shirt_region, mask=cv2.bitwise_not(resized_mask))
            shirt_fg = cv2.bitwise_and(resized_shirt, resized_shirt, mask=resized_mask)

            final_shirt = cv2.add(shirt_bg, shirt_fg)
            frame[shirt_y:shirt_end_y, shirt_x:shirt_end_x] = final_shirt
        except ValueError:
            print("Error: Shirt region out of bounds (still).")

    # Show the frame with the virtual try-on shirt
    cv2.imshow('Virtual Try-On', frame)

    # Break the loop when the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()