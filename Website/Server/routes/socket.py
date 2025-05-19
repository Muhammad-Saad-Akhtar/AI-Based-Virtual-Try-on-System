from flask import Blueprint
from flask_sock import Sock
import numpy as np
import cv2
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


video_socket_route = Blueprint("video_socket_route", __name__)
sock = Sock(video_socket_route)

@sock.route('/ws')
def video_socket(ws):
    import time
    from new import process_frames
    from states import shirt_state
    
    while True:
        data = ws.receive() # receiving data drom the front end.
        if data is None:
            break
        start_time = time.time()
        # Convert bytes to numpy array
        nparr = np.frombuffer(data, np.uint8)   # working on the data
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)   # using the decode function to decode the data

        _, _, processed_frame=process_frames(frame=frame, 
                                shirt_name=shirt_state.shirt_name, 
                                shirt_mask=shirt_state.shirt_mask,
                                shirt_no_bg=shirt_state.shirt_no_bg,
                                fps_history=shirt_state.fps_history)
        # processed_frame = cv2.flip(frame, 1)

        end_time = time.time()
        processing_time = end_time - start_time
        print(f"process_frames took {processing_time:.4f} seconds")

        # Encode back to JPEG
        encode_success, buffer = cv2.imencode('.jpg', processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        
        if not encode_success:
            print("Failed to encode processed frame")
            continue

        ws.send(buffer.tobytes())  # Send processed frame back
