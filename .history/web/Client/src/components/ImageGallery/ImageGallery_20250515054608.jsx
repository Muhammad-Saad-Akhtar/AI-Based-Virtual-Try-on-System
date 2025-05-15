import { useEffect, useState, useRef, useMemo } from "react";
import { io } from "socket.io-client";
import xmarkIcon from "../../assets/Images/xmark-solid.svg";

function ImageGallery({ imageShown = false, setImageShown }) {
  const [images, setImages] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [chosen, setChosen] = useState(null);
  const [tryOnActive, setTryOnActive] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const videoRef = useRef(null);

  // Initialize socket connection
  const socket = useMemo(() => io("http://localhost:5000"), []);

  useEffect(() => {
    // Connect to Python backend when component mounts
    fetch("http://localhost:5000/api/start-session", {
      method: "POST",
    })
      .then((res) => res.json())
      .then((data) => {
        setSessionId(data.session_id);
      })
      .catch((err) => {
        setError("Failed to connect to virtual try-on service");
      });

    // Socket.io event listeners
    socket.on("frame", (data) => {
      if (videoRef.current) {
        const img = new Image();
        img.src = "data:image/jpeg;base64," + data.frame;
        const context = videoRef.current.getContext("2d");
        img.onload = () => {
          context.drawImage(img, 0, 0, videoRef.current.width, videoRef.current.height);
        };
      }
    });

    return () => {
      socket.disconnect();
    };
  }, []);

  useEffect(() => {
    setLoading(true);
    setError(null);

    fetch("http://localhost:9000/api/images")
      .then((res) => {
        if (!res.ok) {
          throw new Error(`Error: ${res.status} - ${res.statusText}`);
        }
        return res.json();
      })
      .then((data) => {
        setImages(data);
        if (data && data.length !== 0) setChosen(data[0]);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  const startVirtualTryOn = async () => {
    if (!chosen || !sessionId) return;

    try {
      // Send selected garment to Python backend
      const formData = new FormData();
      formData.append("session_id", sessionId);
      formData.append("image_path", `${process.env.PUBLIC_URL}/images/${chosen}`);

      await fetch("http://localhost:5000/api/set-garment", {
        method: "POST",
        body: formData,
      });

      setTryOnActive(true);
      socket.emit("start_stream", { session_id: sessionId });
    } catch (err) {
      setError("Failed to start virtual try-on");
    }
  };

  const takeScreenshot = async () => {
    if (!sessionId) return;

    try {
      const formData = new FormData();
      formData.append("session_id", sessionId);

      const response = await fetch("http://localhost:5000/api/save-screenshot", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (data.filename) {
        alert(`Screenshot saved as ${data.filename}`);
      }
    } catch (err) {
      setError("Failed to save screenshot");
    }
  };

  if (loading) {
    return <div>Loading images...</div>;
  }

  if (error) {
    return <div>Error: {error}</div>;
  }

  return (
    <div className="absolute top-0 left-0 w-full h-full z-50 flex justify-center items-center">
      {images.length === 0 ? (
        <div>No images found</div>
      ) : imageShown ? (
        <div className="bg-[#f6f6f6] h-5/6 w-8/12 rounded-2xl flex flex-col">
          <div className="h-[7%] w-full mt-1 flex justify-between items-center content-center px-5">
            <div className="flex gap-4">
              <button
                onClick={startVirtualTryOn}
                className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
              >
                Try On Selected
              </button>
              {tryOnActive && (
                <button
                  onClick={takeScreenshot}
                  className="bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600"
                >
                  Take Screenshot
                </button>
              )}
            </div>
            <img
              onClick={() => {
                setImageShown(false);
                setTryOnActive(false);
              }}
              src={xmarkIcon}
              alt="Close Icon"
              className="w-7 h-auto cursor-pointer"
            />
          </div>
          <div className="h-3/4 w-full grid grid-cols-[4fr_1fr] p-3">
            <div className="bg-[#f6f6f6] flex h-full w-full justify-center items-center p-3 border-2 rounded-2xl">
              {tryOnActive ? (
                <canvas
                  ref={videoRef}
                  width={800}
                  height={600}
                  className="w-full h-auto"
                />
              ) : chosen ? (
                <img
                  src={`http://localhost:9000/images/${chosen}`}
                  alt={`Image`}
                  className="w-1/2 h-auto rounded-lg shadow-lg"
                />
              ) : null}
            </div>
            <div className="bg-[#f6f6f6] flex h-full w-full flex-col overflow-scroll scrollbar-hidden ml-1">
              {images.map((img, idx) => (
                <div
                  key={idx}
                  className="p-6 border-2 m-0.5 rounded-2xl"
                  onClick={() => setChosen(img)}
                >
                  <img
                    src={`http://localhost:9000/images/${img}`}
                    alt={`img-${idx}`}
                    className="w-72 h-auto rounded-lg shadow-lg"
                  />
                </div>
              ))}
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}

export default ImageGallery;
