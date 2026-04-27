import React, { useState } from 'react';

const LABEL_CONFIG = {
  normal: { color: '#22c55e', bg: 'rgba(34,197,94,0.15)', icon: '✅', text: 'Normal Tissue' },
  benign: { color: '#f59e0b', bg: 'rgba(245,158,11,0.15)', icon: '⚠️', text: 'Benign Lesion' },
  malignant: { color: '#ef4444', bg: 'rgba(239,68,68,0.15)', icon: '🚨', text: 'Malignant Lesion' },
  unknown: { color: '#94a3b8', bg: 'rgba(148,163,184,0.15)', icon: '❓', text: 'Unknown' },
};

const ImagePanel = ({ title, src, alt, children }) => {
  const [zoom, setZoom] = useState(false);

  return (
    <div className="result-panel">
      <div className="panel-header">
        <h3 className="panel-title">{title}</h3>
        <button className="btn-zoom" onClick={() => setZoom(!zoom)} title="Toggle zoom">
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
            <path d={zoom
              ? "M3 8H13M8 3V13"
              : "M1 1L6 6M10 10L15 15M15 1L10 6M6 10L1 15"
            } stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
          </svg>
        </button>
      </div>
      {src ? (
        <div className={`panel-image-wrapper ${zoom ? 'zoomed' : ''}`} onClick={() => setZoom(!zoom)}>
          <img src={src} alt={alt} className="panel-image" />
          {children}
        </div>
      ) : (
        <div className="panel-placeholder">
          <div className="placeholder-icon">🔬</div>
          <p>No image available</p>
        </div>
      )}
    </div>
  );
};

const ConfidenceMeter = ({ confidence, label }) => {
  const config = LABEL_CONFIG[label] || LABEL_CONFIG.unknown;
  const pct = Math.round(confidence * 100);

  return (
    <div className="confidence-meter">
      <div className="confidence-header">
        <span className="confidence-label">Confidence</span>
        <span className="confidence-value" style={{ color: config.color }}>{pct}%</span>
      </div>
      <div className="confidence-track">
        <div
          className="confidence-fill"
          style={{
            width: `${pct}%`,
            background: `linear-gradient(90deg, ${config.color}88, ${config.color})`,
          }}
        ></div>
      </div>
    </div>
  );
};

const Result = ({ result }) => {
  const [expanded, setExpanded] = useState(true);

  if (!result) return null;

  const config = LABEL_CONFIG[result.label] || LABEL_CONFIG.unknown;

  return (
    <section id="results" className="section">
      <div className="section-header">
        <div className="section-badge">Step 2</div>
        <h2 className="section-title">Analysis Results</h2>
        <p className="section-subtitle">
          AI-powered segmentation and classification complete
        </p>
      </div>

      {/* Prediction Summary Card */}
      <div className="prediction-summary" style={{ borderColor: config.color, background: config.bg }} id="prediction-summary">
        <div className="prediction-icon" style={{ fontSize: '2.5rem' }}>{config.icon}</div>
        <div className="prediction-main">
          <div className="prediction-label" style={{ color: config.color }}>
            {config.text}
          </div>
          <div className="prediction-filename">
            📁 {result.filename}
          </div>
          <ConfidenceMeter confidence={result.confidence} label={result.label} />
        </div>
        <div className="prediction-stats">
          <div className="stat-item">
            <div className="stat-value">{result.mask_coverage?.toFixed(1)}%</div>
            <div className="stat-label">Area Coverage</div>
          </div>
          <div className="stat-item">
            <div className="stat-value">{result.has_lesion ? 'Yes' : 'No'}</div>
            <div className="stat-label">Lesion Found</div>
          </div>
          <div className="stat-item">
            <div className="stat-value" style={{ color: config.color }}>
              {(result.confidence * 100).toFixed(1)}%
            </div>
            <div className="stat-label">Confidence</div>
          </div>
        </div>
      </div>

      {/* Image Grid */}
      <div className="results-grid" id="results-grid">
        <ImagePanel title="Original Ultrasound" src={result.original_image} alt="Original">
          <div className="image-badge">
            <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
              <circle cx="6" cy="6" r="5" stroke="#a78bfa" strokeWidth="1.5"/>
            </svg>
            Input
          </div>
        </ImagePanel>

        <ImagePanel title="Segmentation Mask" src={result.mask_image} alt="Mask">
          <div className="image-badge" style={{ background: 'rgba(255,255,255,0.15)' }}>
            <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
              <circle cx="6" cy="6" r="5" fill="#6b7280"/>
            </svg>
            Binary Mask
          </div>
        </ImagePanel>

        <ImagePanel title="Overlay Result" src={result.overlay_image} alt="Overlay">
          <div className="image-badge" style={{ background: 'rgba(239,68,68,0.3)' }}>
            <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
              <circle cx="6" cy="6" r="5" fill="#ef4444"/>
            </svg>
            Highlighted Region
          </div>
        </ImagePanel>
      </div>

      {/* Explanation */}
      <div className="explanation-card" id="explanation-card">
        <div className="explanation-header">
          <div className="explanation-icon">📋</div>
          <h3 className="explanation-title">AI Clinical Explanation</h3>
          <button className="btn-expand" onClick={() => setExpanded(!expanded)}>
            {expanded ? '▲ Collapse' : '▼ Expand'}
          </button>
        </div>
        {expanded && (
          <div className="explanation-body">
            <p className="explanation-text">{result.explanation}</p>
            <div className="explanation-disclaimer">
              <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                <circle cx="7" cy="7" r="6" stroke="#f59e0b" strokeWidth="1.5"/>
                <path d="M7 4V7M7 10H7.01" stroke="#f59e0b" strokeWidth="1.5" strokeLinecap="round"/>
              </svg>
              <span>This analysis is AI-generated and should not replace professional medical evaluation.</span>
            </div>
          </div>
        )}
      </div>
    </section>
  );
};

export default Result;
