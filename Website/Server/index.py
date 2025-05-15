from flask import Flask, send_from_directory, Response
from flask_cors import CORS
import os
from routes.routes import image_routes
from check import frame_generator

app = Flask(__name__)
CORS(app)  # Allow React frontend to connect

# Path to your images directory
IMAGES_DIR = os.path.join(os.path.dirname(__file__), 'assets', 'Images')

# Serve static images
@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(IMAGES_DIR, filename)

app.register_blueprint(image_routes, url_prefix='/images')

@app.route("/video-feed")
def video_feed():
    # tell the browser it’s a multipart MJPEG stream
    return Response(
        (b'--frame\r\n'
         b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
         for frame in frame_generator()),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=9000, debug=True)
