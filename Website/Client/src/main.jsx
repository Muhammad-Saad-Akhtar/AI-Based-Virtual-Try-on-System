import { createRoot } from "react-dom/client";
import "./index.css";
import App from "./App.jsx";

createRoot(document.getElementById("root")).render(
  <div className="w-screen h-screen flex flex-col items-center justify-center bg-[linear-gradient(to_bottom_right,_#412d6e_0%,_#4f3980_20%,_#5b458c_40%,_#6a549c_60%,_#7e68b0_70%,_#c8bee6_100%)]">
    <App />
  </div>
);
