import React, { useState, useEffect, useCallback } from 'react';
import Navbar from '../components/Navbar';
import Upload from '../components/Upload';
import Result from '../components/Result';
import History from '../components/History';
import { healthCheck, predictImage, getHistory, clearHistory } from '../services/api';

const Home = () => {
  const [apiStatus, setApiStatus] = useState('loading');
  const [isLoading, setIsLoading] = useState(false);
  const [isClearing, setIsClearing] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [error, setError] = useState(null);

  // Check API health on mount
  useEffect(() => {
    const checkHealth = async () => {
      try {
        await healthCheck();
        setApiStatus('online');
      } catch {
        setApiStatus('offline');
      }
    };
    checkHealth();
  }, []);

  // Fetch history on mount
  const fetchHistory = useCallback(async () => {
    try {
      const data = await getHistory();
      setHistory(data.items || []);
    } catch (err) {
      console.error('Failed to fetch history:', err);
    }
  }, []);

  useEffect(() => {
    fetchHistory();
  }, [fetchHistory]);

  const handlePredict = async (file) => {
    setIsLoading(true);
    setError(null);
    setUploadProgress(0);
    setResult(null);

    try {
      const data = await predictImage(file, setUploadProgress);
      setResult(data);
      // Scroll to results
      setTimeout(() => {
        const el = document.getElementById('results');
        if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }, 100);
      // Refresh history
      await fetchHistory();
    } catch (err) {
      const errMsg = err.response?.data?.detail || err.message || 'Prediction failed';
      setError(errMsg);
      console.error('Prediction error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleClearHistory = async () => {
    setIsClearing(true);
    try {
      await clearHistory();
      setHistory([]);
    } catch (err) {
      console.error('Clear history error:', err);
      setError('Failed to clear history');
    } finally {
      setIsClearing(false);
    }
  };

  return (
    <div className="app">
      <Navbar apiStatus={apiStatus} />

      {/* Hero Section */}
      <div className="hero">
        <div className="hero-content">
          <div className="hero-badge">
            <span className="hero-badge-dot"></span>
            AI-Powered Medical Imaging
          </div>
          <h1 className="hero-title">
            Breast Cancer Detection
            <span className="hero-title-gradient"> AI Platform</span>
          </h1>
          <p className="hero-subtitle">
            Advanced Attention U-Net segmentation combined with CNN classification
            for precise breast ultrasound analysis.
          </p>
          <div className="hero-stats">
            <div className="hero-stat">
              <div className="hero-stat-value">3</div>
              <div className="hero-stat-label">Classes</div>
            </div>
            <div className="hero-stat-divider"></div>
            <div className="hero-stat">
              <div className="hero-stat-value">U-Net</div>
              <div className="hero-stat-label">Architecture</div>
            </div>
            <div className="hero-stat-divider"></div>
            <div className="hero-stat">
              <div className="hero-stat-value">AI</div>
              <div className="hero-stat-label">Powered</div>
            </div>
          </div>
        </div>
        <div className="hero-visual">
          <div className="hero-orb hero-orb-1"></div>
          <div className="hero-orb hero-orb-2"></div>
          <div className="hero-orb hero-orb-3"></div>
        </div>
      </div>

      {/* Error Banner */}
      {error && (
        <div className="error-banner" id="error-banner">
          <div className="error-inner">
            <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
              <circle cx="10" cy="10" r="9" stroke="#ef4444" strokeWidth="1.5"/>
              <path d="M10 6V10M10 14H10.01" stroke="#ef4444" strokeWidth="1.5" strokeLinecap="round"/>
            </svg>
            <span>{error}</span>
            <button className="error-close" onClick={() => setError(null)}>✕</button>
          </div>
        </div>
      )}

      {/* Main Content */}
      <main className="main-content">
        <Upload
          onPredict={handlePredict}
          isLoading={isLoading}
          uploadProgress={uploadProgress}
        />

        {result && (
          <Result result={result} />
        )}

        <History
          items={history}
          onClear={handleClearHistory}
          isLoading={isClearing}
        />
      </main>

      {/* Footer */}
      <footer className="footer">
        <div className="footer-inner">
          <div className="footer-brand">BreastAI</div>
          <div className="footer-text">
            AI-assisted analysis — not a substitute for clinical evaluation
          </div>
          <div className="footer-note">
            Powered by Attention U-Net + CNN | BUSI Dataset
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Home;
