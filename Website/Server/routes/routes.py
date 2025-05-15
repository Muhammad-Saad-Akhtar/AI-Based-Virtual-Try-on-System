from flask import Blueprint, jsonify
import os
import re

# Define the blueprint
image_routes = Blueprint('image_routes', __name__)

# Path to the images directory
IMAGES_DIR = os.path.join(os.path.dirname(__file__), '..', 'assets', 'Images')

# Route to get list of image files
@image_routes.route('/get', methods=['GET'])
def get_images():
    try:
        files = os.listdir(IMAGES_DIR)
        image_files = [f for f in files if re.search(r'\.(jpg|jpeg|png|gif)$', f, re.IGNORECASE)]
        return jsonify(image_files)
    except Exception as e:
        print(f"Error reading image directory: {e}")
        return jsonify({'error': 'Failed to read image folder'}), 500
