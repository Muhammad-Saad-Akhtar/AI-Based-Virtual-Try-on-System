from flask import Blueprint, jsonify, Response, send_from_directory
from check import frame_generator
import os
import re

# Define the blueprint
image_routes = Blueprint('image_routes', __name__)
video_feed_route = Blueprint('video_feed_route', __name__)
chosen_image = Blueprint('chosen_image', __name__)
from states import path_states



# Route to get list of image files
@image_routes.route('/get', methods=['GET'])
def get_images():
    try:
        files = os.listdir(path_states.IMAGES_DIR)
        image_files = [f for f in files if re.search(r'\.(jpg|jpeg|png|gif)$', f, re.IGNORECASE)]
        return jsonify(image_files)
    except Exception as e:
        print(f"Error reading image directory: {e}")
        return jsonify({'error': 'Failed to read image folder'}), 500

@chosen_image.route('/<path:filename>')
def serve_image(filename):
    return send_from_directory(path_states.IMAGES_DIR, filename)

@video_feed_route.route("/video-feed")
def video_feed():
    # tell the browser it’s a multipart MJPEG stream
    return Response(
        (b'--frame\r\n'
         b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
         for frame in frame_generator()),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )
