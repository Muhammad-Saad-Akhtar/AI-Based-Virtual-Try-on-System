# camera_script.py
import cv2

def frame_generator():
    """Yield JPEG-encoded camera frames indefinitely."""
    cap = cv2.VideoCapture(0)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # encode as JPEG
            ret, jpeg = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            yield jpeg.tobytes()
    finally:
        cap.release()
