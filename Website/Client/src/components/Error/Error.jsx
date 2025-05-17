import "./Error.css";

const ErrorLog = ({
  errorText = "Some text",
  refrenceError = "a is not defined",
}) => {
  return (
    <div className="card">
      <div className="header">
        <div className="btn" />
        <div className="btn" />
        <div className="btn" />
        <div className="active">JS console...</div>
      </div>
      <div className="content">
        <div className="res error">
          <span>ReferenceError</span>
          {"{"}" <span>{refrenceError} </span>"{"}"}
        </div>
        <div className="req">console.log(error);</div>
        <div className="res">{errorText}</div>
      </div>
    </div>
  );
};

export default ErrorLog;
