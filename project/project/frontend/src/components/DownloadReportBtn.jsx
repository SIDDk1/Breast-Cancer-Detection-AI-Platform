/**
 * Reusable "Download Report" button.
 *
 * Props:
 *   data  — object with all fields the backend expects (label, confidence, etc.)
 *   size  — 'sm' | 'md' (default 'md')
 */
import React, { useState } from 'react';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const STATES = {
  idle:    { label: 'Download Report', class: '' },
  loading: { label: 'Generating…',     class: 'dl-loading' },
  success: { label: 'Downloaded ✓',    class: 'dl-success' },
  error:   { label: 'Failed — Retry',  class: 'dl-error'   },
};

const DownloadReportBtn = ({ data, size = 'md' }) => {
  const [state, setState] = useState('idle');

  const handleDownload = async (e) => {
    e.stopPropagation(); // prevent parent expand/collapse
    if (state === 'loading') return;

    // Basic validation
    if (!data || !data.label || data.label === 'unknown') {
      setState('error');
      setTimeout(() => setState('idle'), 3000);
      return;
    }

    setState('loading');

    try {
      const payload = {
        record_id:     data._id || data.id || null,
        filename:      data.filename      || 'unknown',
        label:         data.label         || 'unknown',
        confidence:    data.confidence    || 0,
        has_lesion:    data.has_lesion    ?? false,
        mask_coverage: data.mask_coverage != null
                         ? (data.mask_coverage > 1 ? data.mask_coverage / 100 : data.mask_coverage)
                         : 0,
        timestamp:     data.timestamp     || new Date().toISOString(),
        explanation:   data.explanation   || '',
        status:        data.status        || 'success',
        // Prefer stored URL paths (from history); fall back to data: URLs (live result)
        input_image:   data.input_image_url  || data.input_image   || null,
        mask_image:    data.mask_image_url   || data.mask_image    || null,
        overlay_image: data.overlay_image_url|| data.overlay_image || null,
      };

      const res = await fetch(`${API_BASE}/api/report/generate`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(payload),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }

      // Stream → Blob → auto-download
      const blob = await res.blob();
      const url  = URL.createObjectURL(blob);
      const a    = document.createElement('a');
      const ts   = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
      a.href     = url;
      a.download = `Medical_Report_${ts}.pdf`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

      setState('success');
      setTimeout(() => setState('idle'), 3000);
    } catch (err) {
      console.error('Report download error:', err);
      setState('error');
      setTimeout(() => setState('idle'), 4000);
    }
  };

  const s = STATES[state] || STATES.idle;

  return (
    <button
      className={`btn-download-report btn-download-report--${size} ${s.class}`}
      onClick={handleDownload}
      disabled={state === 'loading'}
      title="Download PDF Medical Report"
      id="btn-download-report"
    >
      {state === 'loading' ? (
        <>
          <span className="dl-spinner" />
          <span>{s.label}</span>
        </>
      ) : (
        <>
          {/* Download icon */}
          <svg width="14" height="14" viewBox="0 0 16 16" fill="none"
               className="dl-icon" aria-hidden="true">
            <path d="M8 1v9M4 7l4 4 4-4" stroke="currentColor" strokeWidth="1.8"
                  strokeLinecap="round" strokeLinejoin="round"/>
            <path d="M2 13h12" stroke="currentColor" strokeWidth="1.8"
                  strokeLinecap="round"/>
          </svg>
          <span>{s.label}</span>
        </>
      )}
    </button>
  );
};

export default DownloadReportBtn;
