import React, { useState } from 'react';
import DownloadReportBtn from './DownloadReportBtn';

const LABEL_COLORS = {
  normal: '#22c55e',
  benign: '#f59e0b',
  malignant: '#ef4444',
  unknown: '#94a3b8',
};

const LABEL_ICONS = {
  normal: '✅',
  benign: '⚠️',
  malignant: '🚨',
  unknown: '❓',
};

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';

/** Resolves stored relative paths like /images/... or /outputs/... to full URLs */
const resolveImageUrl = (src) => {
  if (!src) return null;
  if (src.startsWith('data:') || src.startsWith('http://') || src.startsWith('https://')) {
    return src;
  }
  return `${API_BASE}${src}`;
};

const HistoryImageSlot = ({ label, src, icon }) => {
  const [failed, setFailed] = useState(false);
  const url = resolveImageUrl(src);
  const unavailable = !url || failed;

  return (
    <div className="history-img-wrap">
      <div className="history-img-label">{label}</div>
      {unavailable ? (
        <div className="history-img-placeholder">
          <span className="history-img-placeholder-icon">{icon}</span>
          <span className="history-img-placeholder-text">Not Available</span>
        </div>
      ) : (
        <img
          src={url}
          alt={label}
          className="history-img"
          onError={() => setFailed(true)}
        />
      )}
    </div>
  );
};

const HistoryItem = ({ item, index }) => {
  const [expanded, setExpanded] = useState(false);
  const labelColor = LABEL_COLORS[item.label] || LABEL_COLORS.unknown;
  const labelIcon = LABEL_ICONS[item.label] || LABEL_ICONS.unknown;

  const fmtDate = (ts) => {
    if (!ts) return 'Unknown time';
    try {
      const d = new Date(ts);
      return d.toLocaleString('en-US', {
        month: 'short',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
      });
    } catch {
      return ts;
    }
  };

  const thumbUrl = resolveImageUrl(item.overlay_image || item.input_image);

  return (
    <div className={`history-item ${expanded ? 'expanded' : ''}`} id={`history-item-${index}`}>
      <div className="history-item-main" onClick={() => setExpanded(!expanded)}>
        {/* Thumbnail */}
        <div className="history-thumb-wrap">
          {thumbUrl ? (
            <img
              src={thumbUrl}
              alt="result"
              className="history-thumb"
              onError={(e) => { e.target.style.display = 'none'; }}
            />
          ) : (
            <div className="history-thumb-placeholder">🔬</div>
          )}
        </div>

        {/* Info */}
        <div className="history-info">
          <div className="history-label-row">
            <span className="history-icon">{labelIcon}</span>
            <span className="history-label" style={{ color: labelColor }}>
              {item.label?.charAt(0).toUpperCase() + item.label?.slice(1) || 'Unknown'}
            </span>
            <span className="history-confidence">
              {item.confidence ? `${(item.confidence * 100).toFixed(1)}%` : '—'}
            </span>
          </div>
          <div className="history-filename">{item.filename || 'Unknown file'}</div>
          <div className="history-time">{fmtDate(item.timestamp)}</div>
        </div>

        {/* Stats */}
        <div className="history-stats">
          <div className="history-stat">
            <span className="hstat-value">{item.mask_coverage ? (item.mask_coverage * 100).toFixed(1) : '0'}%</span>
            <span className="hstat-label">Coverage</span>
          </div>
          <div className="history-stat">
            <span className="hstat-value" style={{ color: item.has_lesion ? '#ef4444' : '#22c55e' }}>
              {item.has_lesion ? 'Yes' : 'No'}
            </span>
            <span className="hstat-label">Lesion</span>
          </div>
          <div className="history-stat">
            <span className={`hstat-status ${item.status}`}>{item.status || 'success'}</span>
            <span className="hstat-label">Status</span>
          </div>
        </div>

        <div className="history-expand-btn">
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none"
            style={{ transform: expanded ? 'rotate(180deg)' : 'rotate(0deg)', transition: 'transform 0.2s' }}>
            <path d="M4 6L8 10L12 6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </div>
      </div>

      {expanded && (
        <div className="history-detail">
          <div className="history-images-grid">
            <HistoryImageSlot label="Original" src={item.input_image}   icon="🖼️" />
            <HistoryImageSlot label="Mask"     src={item.mask_image}    icon="🎭" />
            <HistoryImageSlot label="Overlay"  src={item.overlay_image} icon="🔍" />
          </div>
          {item.explanation && (
            <div className="history-explanation">{item.explanation}</div>
          )}
          <div className="history-detail-actions">
            <DownloadReportBtn data={item} size="sm" />
          </div>
        </div>
      )}
    </div>
  );
};

const History = ({ items, onClear, isLoading }) => {
  const [confirmClear, setConfirmClear] = useState(false);

  const handleClear = async () => {
    if (!confirmClear) {
      setConfirmClear(true);
      setTimeout(() => setConfirmClear(false), 3000);
      return;
    }
    setConfirmClear(false);
    await onClear();
  };

  return (
    <section id="history" className="section">
      <div className="section-header">
        <div className="section-badge">History</div>
        <h2 className="section-title">Analysis History</h2>
        <p className="section-subtitle">
          {items.length > 0
            ? `${items.length} previous analysis record${items.length > 1 ? 's' : ''} stored in database`
            : 'No analysis history yet. Upload an image to get started.'
          }
        </p>
      </div>

      {items.length > 0 && (
        <div className="history-header-row">
          <div className="history-count-badge">
            <span>{items.length} Records</span>
          </div>
          <button
            className={`btn-clear-history ${confirmClear ? 'confirming' : ''}`}
            onClick={handleClear}
            disabled={isLoading}
            id="btn-clear-history"
          >
            {isLoading ? (
              <>
                <div className="spinner-sm"></div>
                Clearing...
              </>
            ) : confirmClear ? (
              <>
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                  <path d="M3 8L6 11L13 4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
                Confirm Clear
              </>
            ) : (
              <>
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                  <path d="M2 4H14M5 4V2H11V4M6 7V12M10 7V12M3 4L4 14H12L13 4H3Z"
                    stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
                Clear History
              </>
            )}
          </button>
        </div>
      )}

      <div className="history-list" id="history-list">
        {items.length === 0 ? (
          <div className="history-empty">
            <div className="history-empty-icon">📂</div>
            <div className="history-empty-title">No analysis history</div>
            <div className="history-empty-sub">Upload an ultrasound image above to get started</div>
          </div>
        ) : (
          items.map((item, index) => (
            <HistoryItem key={item._id || index} item={item} index={index} />
          ))
        )}
      </div>
    </section>
  );
};

export default History;
