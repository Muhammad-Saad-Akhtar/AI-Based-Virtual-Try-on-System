import React, { useRef, useEffect } from "react";

function StreamProcessor() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const processedRef = useRef(null);
  const socketRef = useRef(null);

  useEffect(() => {
    // Open front camera
    navigator.mediaDevices
      .getUserMedia({ video: { facingMode: "user" }, audio: false })
      .then((stream) => {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      });

    // Connect to WebSocket backend
    socketRef.current = new WebSocket("ws://localhost:9000/ws");

    socketRef.current.onmessage = (event) => {
      const imgBlob = new Blob([event.data], { type: "image/jpeg" });
      const url = URL.createObjectURL(imgBlob);
      processedRef.current.src = url;
    };

    return () => socketRef.current?.close();
  }, []);

  useEffect(() => {
    const interval = setInterval(() => {
      const canvas = canvasRef.current;
      const video = videoRef.current;

      if (
        !canvas ||
        !video ||
        !socketRef.current ||
        socketRef.current.readyState !== 1
      )
        return;

      const ctx = canvas.getContext("2d");
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      canvas.toBlob(
        (blob) => {
          if (blob) {
            socketRef.current.send(blob); // Send JPEG frame
          }
        },
        "image/jpeg",
        0.5
      ); // Adjust quality if needed
    }, 16); // ~10 FPS

    return () => clearInterval(interval);
  }, []);

  return (
    <div>
      <div className="hidden">
        <video ref={videoRef} autoPlay muted style={{ width: "45%" }} />
      </div>
      <canvas ref={canvasRef} style={{ display: "none" }} />
      <img
        ref={processedRef}
        alt="Processed Stream"
        style={{ width: "100%" }}
      />
    </div>
  );
}

export default StreamProcessor;
