import { useState } from "react";
import "./App.css";
import GlowButton from "./components/GlowButton/GlowButton";
import ImageGallery from "./components/ImageGallery/ImageGallery";
import CameraFeed from "./components/CameraFeed/CameraFeed";
import Loader from "./components/loader/loader";

function App() {
  const [imageShown, setimageShown] = useState(false);
  const [imageState, setImageState] = useState("not-prepared");
  const [videoShown, setVideoShown] = useState(false);

  if (imageState === "loading") return <Loader />;
  // else if (videoShown) return <StreamProcessor />;
  else if (videoShown) return <CameraFeed />;
  else
    return (
      <>
        {imageShown ? (
          <ImageGallery
            imageShown={imageShown}
            setImageShown={setimageShown}
            setImageState={setImageState}
          />
        ) :
        null}
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
        <div className="fade-in-up-last content-center justify-center mb-6">
          <GlowButton check={imageShown} setCheck={setimageShown} />
        </div>
        {imageState === "prepared" ? (
          <div className="fade-in-up-last content-center justify-center">
            <GlowButton
            buttonText="Start Stream"
              onClickHandle={() => {
                setImageState("not-prepared");
                setVideoShown(true);
              }}
            />
          </div>
        ) : null}
      </>
    );
}

export default App;
