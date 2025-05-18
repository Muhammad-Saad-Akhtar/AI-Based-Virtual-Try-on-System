from flask import Flask, Blueprint, jsonify
from flask_sock import Sock
import numpy as np
import cv2

video_socket_route = Blueprint("video_socket_route", __name__)
sock = Sock(video_socket_route)

@sock.route('/ws')
def video_socket(ws):
    while True:
        data = ws.receive() # receiving data drom the front end.
        if data is None:
            break

        # Convert bytes to numpy array
        nparr = np.frombuffer(data, np.uint8)   # working on the data
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)   # using the decode function to decode the data

        #? Apply OpenCV processing
        processed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        inverted = cv2.flip(processed, 1)


        # Encode back to JPEG
        _, buffer = cv2.imencode('.jpg', inverted)  # encoding the data 
        ws.send(buffer.tobytes())  # Send processed frame back
