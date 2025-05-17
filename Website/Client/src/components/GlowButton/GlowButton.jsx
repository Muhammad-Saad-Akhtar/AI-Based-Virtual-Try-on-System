import "./GlowButton.css";

const GlowButton = ({ buttonText = "Select a shirt", check, setCheck, onClickHandle = () => setCheck(!check) }) => {
  return <button onClick={onClickHandle}>{buttonText}</button>;
};

export default GlowButton;
