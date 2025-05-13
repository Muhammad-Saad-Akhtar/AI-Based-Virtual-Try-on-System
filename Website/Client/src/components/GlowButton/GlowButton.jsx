import "./GlowButton.css";

const GlowButton = ({ check, setCheck }) => {
  return <button onClick={() => setCheck(!check)}>Select a shirt</button>;
};

export default GlowButton;
