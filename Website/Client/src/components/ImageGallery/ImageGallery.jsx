import { useEffect, useState } from "react";
import xmarkIcon from "../../assets/Images/xmark-solid.svg";
import Loader from "../loader/loader";
import ErrorLog from "../Error/Error";

function ImageGallery({ imageShown = false, setImageShown, setImageState }) {
  const [images, setImages] = useState([]);
  const [loading, setLoading] = useState(true); // Track loading state
  const [error, setError] = useState(null); // Track any errors
  const [chosen, setChosen] = useState(null); // The chosen image]

  const prepareShirt = async (shirtImagePath, setImageShown, setImageState) => {
    try {
      console.log("Runing prepare");
      setImageShown(false);
      setImageState("loading"); // Set state to loading before request

      const response = await fetch("http://localhost:9000/prepare-shirt", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          shirt_image_path: shirtImagePath,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        setImageState("not-prepared");
        return { error: data.error || "Server error" };
      }

      setImageState("prepared"); // Successfully prepared
      setImageShown(false);
      return data;
    } catch (error) {
      setImageState("not-prepared"); // On network or fetch failure
      return { error: "Network error or server not reachable" };
    }
  };

  useEffect(() => {
    // Reset error and loading states on each fetch attempt
    setLoading(true);
    setError(null);

    fetch("http://localhost:9000/images/get")
      .then((res) => {
        if (!res.ok) {
          // If the response is not okay, throw an error
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
        // Catch any error (network or response-related) and set it to state
        setError(err.message);
        setLoading(false);
      });
  }, []); // Empty dependency array means this runs once when the component mounts

  if (loading) {
    return (
      <div className="absolute top-0 left-0 w-full h-full z-50 flex justify-center items-center">
        <Loader />
      </div>
    );
  }

  if (error) {
    return (
      <div className="absolute top-0 left-0 w-full h-full z-50 flex justify-center items-center">
        <ErrorLog errorText={error} refrenceError="Error while fetching" />
      </div>
    );
  }

  return (
    <div className="absolute top-0 left-0 w-full h-full z-50 flex justify-center items-center">
      {images.length === 0 ? (
        <div className="absolute top-0 left-0 w-full h-full z-50 flex justify-center items-center">
          <ErrorLog
            errorText={"No Images found"}
            refrenceError="Image length = 0"
          />
        </div>
      ) : imageShown ? (
        <div className="bg-[#f6f6f6] h-5/6 w-8/12 rounded-2xl flex flex-col">
          <div className="h-[7%] w-full mt-1 flex justify-end-safe items-center content-center mr-5">
            <img
              onClick={() => setImageShown(false)}
              src={xmarkIcon}
              alt="Close Icon"
              className="w-7 h-auto cursor-pointer mr-5 mt-2"
            />
          </div>
          <div className="h-3/4 w-full grid grid-cols-[4fr_1fr] p-3">
            <div
              onClick={() => {}}
              className="bg-[#f6f6f6] flex h-full w-full justify-center items-center p-3 border-2 rounded-2xl"
            >
              {chosen ? (
                <img
                  onClick={() => {
                    console.log(chosen);
                    prepareShirt(chosen, setImageShown, setImageState);
                  }}
                  src={`http://localhost:9000/images/${chosen}`}
                  alt={`Image`}
                  className="w-1/2 h-auto rounded-lg shadow-lg cursor-pointer"
                />
              ) : null}
            </div>
            <div className="bg-[#f6f6f6] flex h-full w-full flex-col overflow-scroll scrollbar-hidden ml-1">
              {images.map((img, idx) => {
                return (
                  <div
                    className="p-6 border-2 m-0.5 rounded-2xl"
                    onClick={() => setChosen(img)}
                  >
                    <img
                      key={idx}
                      src={`http://localhost:9000/images/${img}`}
                      alt={`img-${idx}`}
                      className="w-72 h-auto rounded-lg shadow-lg cursor-pointer"
                    />
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}

export default ImageGallery;
