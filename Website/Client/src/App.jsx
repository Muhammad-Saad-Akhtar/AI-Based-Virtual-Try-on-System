import { useState } from "react";
import "./App.css";
import GlowButton from "./components/GlowButton/GlowButton";
import ImageGallery from "./components/ImageGallery/ImageGallery";
import SlideButton from "./components/SlideButton/SlideButton";
import CameraFeed from "./components/CameraFeed/CameraFeed";
import StreamProcessor from "./components/SocketCameraFeed/SocketCameraFeed";

function App() {
  const [imageShown, setimageShown] = useState(false);

  return (
    <div className="w-screen h-screen flex flex-col items-center justify-center bg-[linear-gradient(to_bottom_right,_#412d6e_0%,_#4f3980_20%,_#5b458c_40%,_#6a549c_60%,_#7e68b0_70%,_#c8bee6_100%)]">
      {imageShown ? (
        // <ImageGallery imageShown={imageShown} setImageShown={setimageShown} />
        // <CameraFeed />
        <StreamProcessor />
      ) : null}
      <div className="pb-10">
        <h1 className="fade-in-up font-[pixelify] font-bold text-9xl text-center text-shiny-gradient relative overflow-hidden">
          AI Based Virtual
          <span className="shine metallic" />
        </h1>
        <h1 className="fade-in-up-delayed font-[pixelify] font-extrabold text-9xl text-center text-shiny-gradient relative overflow-hidden">
          Try-On
          <span className="shine metallicDelay" />
        </h1>
      </div>
      <div className="fade-in-up-last content-center justify-center">
        {/* <SlideButton /> */}
        <GlowButton check={imageShown} setCheck={setimageShown} />
      </div>
    </div>
  );
}

export default App;
