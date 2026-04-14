import React, { useState, useRef, useCallback } from 'react';

const Upload = ({ onPredict, isLoading, uploadProgress }) => {
  const [dragOver, setDragOver] = useState(false);
  const [preview, setPreview] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const fileInputRef = useRef();

  const handleFile = useCallback((file) => {
    if (!file) return;
    if (!file.type.startsWith('image/')) {
      alert('Please upload an image file (PNG, JPG, etc.)');
      return;
    }
    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setPreview(url);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    handleFile(file);
  }, [handleFile]);

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = () => setDragOver(false);

  const handleFileInput = (e) => {
    handleFile(e.target.files[0]);
  };

  const handleSubmit = () => {
    if (!selectedFile) {
      alert('Please select an image first.');
      return;
    }
    onPredict(selectedFile);
  };

  const handleClear = () => {
    setPreview(null);
    setSelectedFile(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  return (
    <section id="upload" className="section">
      <div className="section-header">
        <div className="section-badge">Step 1</div>
        <h2 className="section-title">Upload Ultrasound Image</h2>
        <p className="section-subtitle">
          Upload a breast ultrasound image in PNG or JPG format for AI-powered analysis
        </p>
      </div>

      <div className="upload-wrapper">
        {!preview ? (
          <div
            className={`drop-zone ${dragOver ? 'drag-over' : ''}`}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onClick={() => fileInputRef.current?.click()}
            id="drop-zone"
          >
            <div className="drop-zone-icon">
              <svg width="64" height="64" viewBox="0 0 64 64" fill="none">
                <circle cx="32" cy="32" r="31" stroke="url(#uploadGrad)" strokeWidth="1.5" strokeDasharray="4 2"/>
                <path d="M32 20L32 44M22 30L32 20L42 30" stroke="url(#uploadGrad)" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"/>
                <defs>
                  <linearGradient id="uploadGrad" x1="0" y1="0" x2="64" y2="64">
                    <stop offset="0%" stopColor="#a78bfa"/>
                    <stop offset="100%" stopColor="#ec4899"/>
                  </linearGradient>
                </defs>
              </svg>
            </div>
            <p className="drop-zone-title">Drag & drop your image here</p>
            <p className="drop-zone-subtitle">or click to browse files</p>
            <div className="drop-zone-formats">
              <span className="format-tag">PNG</span>
              <span className="format-tag">JPG</span>
              <span className="format-tag">JPEG</span>
              <span className="format-tag">BMP</span>
            </div>
          </div>
        ) : (
          <div className="preview-container">
            <div className="preview-image-wrapper">
              <img src={preview} alt="Preview" className="preview-image" id="preview-image" />
              <div className="preview-overlay">
                <div className="preview-badges">
                  <div className="preview-badge">
                    <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                      <path d="M7 1L9 5H13L10 8L11 12L7 10L3 12L4 8L1 5H5L7 1Z" fill="#22c55e"/>
                    </svg>
                    <span>Image loaded</span>
                  </div>
                  <div className="preview-badge">
                    <span>{selectedFile?.name}</span>
                  </div>
                </div>
              </div>
            </div>
            <div className="preview-actions">
              <button className="btn-secondary" onClick={handleClear} disabled={isLoading} id="btn-clear-image">
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                  <path d="M3 3L13 13M13 3L3 13" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                </svg>
                Clear
              </button>
              <button className="btn-secondary" onClick={() => fileInputRef.current?.click()} disabled={isLoading} id="btn-change-image">
                Change Image
              </button>
            </div>
          </div>
        )}

        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleFileInput}
          style={{ display: 'none' }}
          id="file-input"
        />
      </div>

      {isLoading && (
        <div className="upload-progress">
          <div className="progress-bar-wrapper">
            <div className="progress-bar-fill" style={{ width: `${uploadProgress}%` }}></div>
          </div>
          <div className="progress-text">
            <div className="spinner-sm"></div>
            <span>{uploadProgress < 100 ? `Uploading... ${uploadProgress}%` : 'Running AI analysis...'}</span>
          </div>
        </div>
      )}

      <div className="upload-actions">
        <button
          className={`btn-primary ${!selectedFile || isLoading ? 'btn-disabled' : ''}`}
          onClick={handleSubmit}
          disabled={!selectedFile || isLoading}
          id="btn-analyze"
        >
          {isLoading ? (
            <>
              <div className="spinner-sm"></div>
              <span>Analyzing...</span>
            </>
          ) : (
            <>
              <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
                <circle cx="9" cy="9" r="8" stroke="currentColor" strokeWidth="1.5"/>
                <path d="M9 5V9L12 11" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
              </svg>
              <span>Run AI Analysis</span>
            </>
          )}
        </button>
      </div>

      <div className="upload-info">
        <div className="info-card">
          <div className="info-icon">🔬</div>
          <div>
            <div className="info-title">Attention U-Net Segmentation</div>
            <div className="info-desc">Precisely segments breast lesion regions</div>
          </div>
        </div>
        <div className="info-card">
          <div className="info-icon">🧠</div>
          <div>
            <div className="info-title">CNN Classification</div>
            <div className="info-desc">Classifies as Normal, Benign, or Malignant</div>
          </div>
        </div>
        <div className="info-card">
          <div className="info-icon">⚡</div>
          <div>
            <div className="info-title">Fast Inference</div>
            <div className="info-desc">Results typically in under 5 seconds</div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Upload;
