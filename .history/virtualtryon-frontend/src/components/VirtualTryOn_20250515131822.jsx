import { useEffect, useRef, useState } from 'react';
import io from 'socket.io-client';

const VirtualTryOn = ({ selectedGarment, onBack }) => {
  const [sessionId, setSessionId] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const videoRef = useRef(null);
  const socketRef = useRef(null);

  useEffect(() => {
    // Start a new session when component mounts
    fetch('http://localhost:5000/api/start-session', {
      method: 'POST',
    })
      .then(res => res.json())
      .then(data => {
        setSessionId(data.session_id);
        // Connect to WebSocket after getting session ID
        socketRef.current = io('http://localhost:5000');
        socketRef.current.on('connect', () => {
          setIsConnected(true);
          // Start the video stream
          socketRef.current.emit('start_stream', { session_id: data.session_id });
        });

        // Handle incoming video frames
        socketRef.current.on('frame', (data) => {
          if (videoRef.current) {
            const img = new Image();
            img.onload = () => {
              const canvas = videoRef.current;
              const ctx = canvas.getContext('2d');
              ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            };
            img.src = `data:image/jpeg;base64,${data.frame}`;
          }
        });
      });

    // Set the selected garment
    if (selectedGarment && sessionId) {
      fetch('http://localhost:5000/api/set-garment', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: `session_id=${sessionId}&image_path=${selectedGarment.path}`,
      });
    }

    return () => {
      if (socketRef.current) {
        socketRef.current.disconnect();
      }
    };
  }, [sessionId, selectedGarment]);

  const saveScreenshot = () => {
    if (sessionId) {
      fetch('http://localhost:5000/api/save-screenshot', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: `session_id=${sessionId}`,
      })
        .then(res => res.json())
        .then(data => {
          if (data.filename) {
            alert(`Screenshot saved as ${data.filename}`);
          }
        });
    }
  };

  return (
    <div className="flex flex-col items-center">
      <div className="relative w-full max-w-4xl">
        <canvas
          ref={videoRef}
          width={800}
          height={600}
          className="w-full rounded-lg shadow-xl"
        />
        <div className="absolute top-4 left-4">
          <button
            onClick={onBack}
            className="bg-purple-600 text-white px-4 py-2 rounded-lg hover:bg-purple-700"
          >
            Back to Gallery
          </button>
        </div>
        <div className="absolute top-4 right-4">
          <button
            onClick={saveScreenshot}
            className="bg-green-600 text-white px-4 py-2 rounded-lg hover:bg-green-700"
          >
            Take Screenshot
          </button>
        </div>
      </div>
      {!isConnected && (
        <div className="mt-4 text-white bg-red-500 px-4 py-2 rounded">
          Connecting to server...
        </div>
      )}
    </div>
  );
};

export default VirtualTryOn;
