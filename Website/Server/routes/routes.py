from flask import Blueprint, jsonify, Response, send_from_directory
import sys
import os
import re

# Define the blueprint
image_routes = Blueprint('image_routes', __name__)
video_feed_route = Blueprint('video_feed_route', __name__)
chosen_image = Blueprint('chosen_image', __name__)
from states import path_states

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))


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
    from new import video_generator
    from states import shirt_state

    return Response(
        (b'--frame\r\n'
         b'Content-Type: image/jpeg\r\n\r\n' + frm + b'\r\n'
         for frm in video_generator(shirt_state=shirt_state)),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )
