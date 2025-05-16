import "./GlowButton.css";

const GlowButton = ({ check, setCheck, onClickHandle = () => setCheck(!check) }) => {
  return <button onClick={onClickHandle}>Select a shirt</button>;
};

export default GlowButton;
