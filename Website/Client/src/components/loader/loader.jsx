import "./loader.css"

const Loader = () => {
  return (
      <div className="terminal-loader">
        <div className="terminal-header">
          <div className="terminal-title">Status</div>
          <div className="terminal-controls">
            <div className="control close" />
            <div className="control minimize" />
            <div className="control maximize" />
          </div>
        </div>
        <div className="text">Loading...</div>
      </div>
  );
}


export default Loader;
