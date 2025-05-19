import React, { useRef } from 'react'


export default function CameraFeed() {
  const canvasRef = useRef(null);

  return (
    <div>
      {/* MJPEG feed from Flask */}
      <canvas ref={canvasRef} style={{ display: "none" }} />
      <img
        src="http://localhost:9000/video-feed"
        alt="Live camera feed"
        style={{ width: "100%", maxWidth: 640, border: "1px solid #ccc" }}
      />
    </div>
  );
}

