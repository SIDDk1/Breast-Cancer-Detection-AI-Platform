import React from 'react';

const Navbar = ({ apiStatus }) => {
  return (
    <nav className="navbar">
      <div className="navbar-inner">
        <div className="navbar-brand">
          <div className="brand-icon">
            <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
              <circle cx="14" cy="14" r="13" stroke="url(#grad1)" strokeWidth="2"/>
              <path d="M14 7C14 7 8 12 8 16C8 19.3 10.7 22 14 22C17.3 22 20 19.3 20 16C20 12 14 7 14 7Z" fill="url(#grad2)"/>
              <circle cx="14" cy="16" r="3" fill="white" opacity="0.9"/>
              <defs>
                <linearGradient id="grad1" x1="0" y1="0" x2="28" y2="28">
                  <stop offset="0%" stopColor="#a78bfa"/>
                  <stop offset="100%" stopColor="#ec4899"/>
                </linearGradient>
                <linearGradient id="grad2" x1="0" y1="0" x2="28" y2="28">
                  <stop offset="0%" stopColor="#a78bfa"/>
                  <stop offset="100%" stopColor="#ec4899"/>
                </linearGradient>
              </defs>
            </svg>
          </div>
          <div className="brand-text">
            <span className="brand-name">BreastAI</span>
            <span className="brand-tagline">Intelligent Detection</span>
          </div>
        </div>

        <div className="navbar-center">
          <div className="nav-links">
            <a href="#upload" className="nav-link">Analyze</a>
            <a href="#results" className="nav-link">Results</a>
            <a href="#history" className="nav-link">History</a>
          </div>
        </div>

        <div className="navbar-right">
          <div className={`status-badge ${apiStatus === 'online' ? 'status-online' : apiStatus === 'loading' ? 'status-loading' : 'status-offline'}`}>
            <span className="status-dot"></span>
            <span className="status-text">
              {apiStatus === 'online' ? 'AI Online' : apiStatus === 'loading' ? 'Connecting...' : 'Offline'}
            </span>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
