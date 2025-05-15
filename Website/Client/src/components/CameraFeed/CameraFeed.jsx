import React from 'react'

export default function CameraFeed() {
  return (
    <div>
      <h2>Your Live Camera View</h2>
      {/* MJPEG feed from Flask */}
      <img
        src="http://localhost:9000/video-feed"
        alt="Live camera feed"
        style={{ width: "100%", maxWidth: 640, border: "1px solid #ccc" }}
      />
    </div>
  );
}

