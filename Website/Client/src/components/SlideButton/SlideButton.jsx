import "./SlideButton.css";

const SlideButton = ({ check, setCheck }) => {
  return (
    <button className="btn-17" onClick={() => setCheck(!check)}>
      <span className="text-container">
        <span className="text">Select a shirt</span>
      </span>
    </button>
  );
};

export default SlideButton;
