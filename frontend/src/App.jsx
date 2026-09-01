import { useCallback, useEffect, useRef, useState } from 'react'
import {
  AlertTriangle,
  Bot,
  FileImage,
  ImageIcon,
  User,
  PlayCircle,
  Scan,
  Maximize,
  X,
  Info,
  Share2,
  CheckCircle2,
  Eye,
  ShieldAlert,
  Landmark,
  Phone,
  ExternalLink,
  Zap,
  ShieldCheck,
  Activity,
  ChevronDown,
} from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import './index.css'

/* ── Constants ──────────────────────────────────────────────── */

const CHECKPOINTS = {
  aug: 'Augmented',
  baseline: 'Baseline',
}

const BATCH_LIMIT = 50

/* ── Helpers ────────────────────────────────────────────────── */

function classify(score) {
  if (score == null) return { verdict: 'Unable to analyze', detail: 'This image could not be processed. Please try another.', tone: 'neutral', isAI: false }
  if (score >= 0.7) return { verdict: 'Likely AI-Generated', detail: 'We found strong indicators that this image was synthetically generated.', tone: 'high', isAI: true }
  if (score >= 0.4) return { verdict: 'Inconclusive', detail: 'The image has mixed signals. It may be heavily edited or AI-generated.', tone: 'mid', isAI: false }
  return { verdict: 'Likely Authentic', detail: 'We found very few synthetic indicators. This appears to be a human-made image.', tone: 'low', isAI: false }
}

async function generateShareCard(imageSrc, verdict, score, tone, isAI) {
  return new Promise((resolve, reject) => {
    const canvas = document.createElement('canvas');
    const width = 1080;
    const height = 1920;
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');

    // Rich Dark Gradient Background
    const bgGradient = ctx.createLinearGradient(0, 0, width, height);
    bgGradient.addColorStop(0, '#09090b');
    bgGradient.addColorStop(1, '#18181b');
    ctx.fillStyle = bgGradient;
    ctx.fillRect(0, 0, width, height);

    // Glowing Orbs
    ctx.filter = 'blur(200px)';
    ctx.fillStyle = tone === 'high' ? 'rgba(254, 44, 85, 0.5)' : tone === 'mid' ? 'rgba(245, 166, 35, 0.4)' : 'rgba(37, 244, 238, 0.4)';
    ctx.beginPath(); ctx.arc(300, 400, 600, 0, Math.PI*2); ctx.fill();
    ctx.fillStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.beginPath(); ctx.arc(800, 1500, 500, 0, Math.PI*2); ctx.fill();
    ctx.filter = 'none';

    // HUD Grid pattern
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
    ctx.lineWidth = 1;
    for(let i=0; i<width; i+=60) {
      ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, height); ctx.stroke();
    }
    for(let i=0; i<height; i+=60) {
      ctx.beginPath(); ctx.moveTo(0, i); ctx.lineTo(width, i); ctx.stroke();
    }

    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      // Header Brand
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.font = '800 45px "Inter", sans-serif';
      ctx.fillStyle = '#FFFFFF';
      ctx.letterSpacing = '5px';
      ctx.fillText('IMAGESIGNAL', width/2, 120);
      ctx.letterSpacing = '0px';

      // Fixed Image Box Bounds
      const boxW = 860;
      const boxH = 860;
      const boxX = (width - boxW) / 2;
      const boxY = 220; 

      // Glassmorphic Image Container
      ctx.fillStyle = 'rgba(255, 255, 255, 0.03)';
      if (ctx.roundRect) {
        ctx.beginPath(); ctx.roundRect(boxX, boxY, boxW, boxH, 40); ctx.fill();
      } else {
        ctx.fillRect(boxX, boxY, boxW, boxH);
      }
      
      const accentColor = tone === 'high' ? '#FE2C55' : tone === 'mid' ? '#F5A623' : '#25F4EE';
      ctx.strokeStyle = accentColor;
      ctx.lineWidth = 4;
      if (ctx.roundRect) {
        ctx.beginPath(); ctx.roundRect(boxX, boxY, boxW, boxH, 40); ctx.stroke();
      } else {
        ctx.strokeRect(boxX, boxY, boxW, boxH);
      }

      // HUD Corner Brackets
      const bLen = 60;
      ctx.lineWidth = 8;
      ctx.strokeStyle = '#FFFFFF';
      const drawBracket = (x, y, dx, dy) => {
        ctx.beginPath(); ctx.moveTo(x+dx, y); ctx.lineTo(x, y); ctx.lineTo(x, y+dy); ctx.stroke();
      };
      drawBracket(boxX-20, boxY-20, bLen, bLen);
      drawBracket(boxX+boxW+20, boxY-20, -bLen, bLen);
      drawBracket(boxX-20, boxY+boxH+20, bLen, -bLen);
      drawBracket(boxX+boxW+20, boxY+boxH+20, -bLen, -bLen);

      // Image clipping mask (so image honors border radius)
      ctx.save();
      if (ctx.roundRect) {
        ctx.beginPath(); ctx.roundRect(boxX, boxY, boxW, boxH, 40); ctx.clip();
      }

      // Calculate Image Scaling (cover within box)
      const ratio = Math.max(boxW/img.width, boxH/img.height);
      const imgW = img.width * ratio;
      const imgH = img.height * ratio;
      const imgX = boxX + (boxW - imgW) / 2;
      const imgY = boxY + (boxH - imgH) / 2; 

      ctx.drawImage(img, imgX, imgY, imgW, imgH);
      ctx.restore();

      // Verdict Label
      ctx.font = '800 110px "Inter", sans-serif';
      ctx.fillStyle = accentColor;
      ctx.shadowColor = accentColor;
      ctx.shadowBlur = 30;
      ctx.fillText(verdict.toUpperCase(), width/2, 1260, 960);
      ctx.shadowBlur = 0; // reset

      // Score Text
      ctx.font = '600 40px "JetBrains Mono", monospace';
      ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
      ctx.fillText('CONFIDENCE SCORE', width/2, 1420);
      
      ctx.font = '900 240px "Inter", sans-serif';
      ctx.fillStyle = '#FFFFFF';
      ctx.fillText(`${score}%`, width/2, 1600);

      // Footer
      ctx.font = '500 32px "Inter", sans-serif';
      ctx.fillStyle = 'rgba(255,255,255,0.4)';
      ctx.fillText('Verify authenticity at imagesignal.app', width/2, 1820);

      canvas.toBlob((blob) => resolve(blob), 'image/png');
    };
    img.onerror = reject;
    img.src = imageSrc;
  });
}

/* ── Background Animation ───────────────────────────────────── */

function AnimatedBackground() {
  return (
    <div className="animated-bg">
      <motion.div
        className="blob blob-1"
        animate={{
          x: [0, 100, 0, -100, 0],
          y: [0, 50, 100, 50, 0],
          scale: [1, 1.1, 1, 0.9, 1],
        }}
        transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
      />
      <motion.div
        className="blob blob-2"
        animate={{
          x: [0, -80, 0, 80, 0],
          y: [0, -50, -100, -50, 0],
          scale: [1, 0.9, 1, 1.1, 1],
        }}
        transition={{ duration: 25, repeat: Infinity, ease: "linear" }}
      />
    </div>
  )
}

/* ── ELA Heatmap (Frontend UI Trick) ────────────────────────── */

function ElaHeatmap({ src, fullscreen = false }) {
  const [loading, setLoading] = useState(true);

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      {loading && <div className="spinner" style={{ position: 'absolute', zIndex: 5 }}></div>}
      {src && (
        <img 
          src={src} 
          onLoad={() => setLoading(false)}
          style={{ 
            maxWidth: '100%', 
            maxHeight: fullscreen ? '90vh' : '300px', 
            borderRadius: 'var(--radius-sm)', 
            border: '1px solid var(--border)', 
            boxShadow: 'var(--shadow-md)',
            opacity: loading ? 0 : 1,
            transition: 'opacity 0.3s ease'
          }} 
          alt="ELA Heatmap"
        />
      )}
    </div>
  );
}


/* ── Transition Mascot ──────────────────────────────────────── */

function TransitionBot({ onComplete }) {
  useEffect(() => {
    const timer = setTimeout(() => {
      onComplete();
    }, 500);
    return () => clearTimeout(timer);
  }, [onComplete]);

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '60vh', zIndex: 10 }}>
      <motion.div
        initial={{ y: 50, opacity: 0 }}
        animate={{ y: [0, -15, 0], opacity: 1 }}
        transition={{ y: { duration: 1.5, repeat: Infinity, ease: "easeInOut" }, opacity: { duration: 0.3 } }}
        style={{ width: 140, height: 140, position: 'relative' }}
      >
        <svg viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
          <rect x="20" y="20" width="60" height="45" rx="8" fill="var(--panel-bg)" stroke="var(--text-main)" strokeWidth="4"/>
          <line x1="50" y1="20" x2="50" y2="5" stroke="var(--text-main)" strokeWidth="4"/>
          <circle cx="50" cy="5" r="4" fill="var(--brand-cyan)">
             <animate attributeName="opacity" values="0;1;0" dur="1s" repeatCount="indefinite" />
          </circle>
          <motion.circle cx="35" cy="40" r="5" fill="var(--brand-magenta)"
             animate={{ scaleY: [1, 0.1, 1] }}
             transition={{ duration: 2, repeat: Infinity, times: [0, 0.1, 0.2] }}
          />
          <motion.circle cx="65" cy="40" r="5" fill="var(--brand-magenta)"
             animate={{ scaleY: [1, 0.1, 1] }}
             transition={{ duration: 2, repeat: Infinity, times: [0, 0.1, 0.2] }}
          />
          <rect x="30" y="55" width="40" height="4" rx="2" fill="var(--border)"/>
          <motion.rect x="30" y="55" width="10" height="4" rx="2" fill="var(--brand-cyan)"
             animate={{ x: [0, 30, 0] }}
             transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
          />
        </svg>

        <motion.div
          style={{
            position: 'absolute',
            top: 20,
            left: -20,
            width: 180,
            height: 2,
            background: 'var(--brand-cyan)',
            boxShadow: '0 0 12px var(--brand-cyan)'
          }}
          animate={{ y: [0, 60, 0] }}
          transition={{ duration: 1.2, repeat: Infinity, ease: "linear" }}
        />
      </motion.div>
      <motion.h3
        initial={{ opacity: 0 }}
        animate={{ opacity: [0.5, 1, 0.5] }}
        transition={{ duration: 1, repeat: Infinity }}
        style={{ marginTop: '32px', color: 'var(--text-main)', letterSpacing: '0.15em', fontSize: '13px' }}
      >
        INITIALIZING CORE...
      </motion.h3>
    </div>
  )
}

/* ── Header ─────────────────────────────────────────────────── */

function Header({ theme, onToggleTheme, currentView, setView }) {
  return (
    <div className="top-bar">
      <div className="brand">ImageSignal</div>
      <div className="nav-links">
        <button className={`nav-btn ${currentView === 'home' ? 'active' : ''}`} onClick={() => setView('home')}>Home</button>
        <button className={`nav-btn ${currentView === 'tool' ? 'active' : ''}`} onClick={() => setView('transition')}>Analyzer</button>
        <button className={`nav-btn ${currentView === 'batch' ? 'active' : ''}`} onClick={() => setView('batch')}>Batch</button>
        <button className={`nav-btn ${currentView === 'learn' ? 'active' : ''}`} onClick={() => setView('learn')}>Safety Guide</button>
        
        <div className="nav-dropdown">
          <button className={`nav-btn ${(currentView === 'brief' || currentView === 'api') ? 'active' : ''}`} style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            Developer <ChevronDown size={14} />
          </button>
          <div className="nav-dropdown-menu">
            <button className={`nav-btn ${currentView === 'brief' ? 'active' : ''}`} onClick={() => setView('brief')}>Project Brief</button>
            <button className={`nav-btn ${currentView === 'api' ? 'active' : ''}`} onClick={() => setView('api')}>API Docs</button>
          </div>
        </div>
      </div>
      <button className="theme-btn" onClick={onToggleTheme} title="Toggle Dark Mode">
        {theme === 'dark' ? 'Light Mode' : 'Dark Mode'}
      </button>
    </div>
  )
}


/* ── API Docs View ──────────────────────────────────────────── */

function ApiDocs() {
  return (
    <div className="api-docs-container">
      <div className="api-docs-content">
        <h1>API Documentation</h1>
        <p style={{fontSize: 16, color: 'var(--text-dim)'}}>Integrate ImageSignal's detection engine directly into your own applications using our REST API.</p>
        
        <div className="api-endpoint">
          <div className="endpoint-header">
            <span className="method">POST</span>
            <span className="path">/api/analyze</span>
          </div>
          <p>Analyzes a single image and returns probability scores, compression metrics, and visual previews.</p>
          
          <h3 style={{marginTop: 32}}>Parameters (FormData)</h3>
          <table className="api-table">
            <thead>
              <tr><th>Name</th><th>Type</th><th>Description</th></tr>
            </thead>
            <tbody>
              <tr>
                <td><code>image</code></td>
                <td>File</td>
                <td>The image file to analyze (Max 25MB). Supported: JPG, PNG, WebP, BMP.</td>
              </tr>
              <tr>
                <td><code>checkpoint</code></td>
                <td>String</td>
                <td>Model checkpoint to use. Options: <code>aug</code> (default), <code>baseline</code>.</td>
              </tr>
              <tr>
                <td><code>quality</code></td>
                <td>Integer</td>
                <td>JPEG quality level for re-encoding step (30-95). Default: <code>95</code>.</td>
              </tr>
              <tr>
                <td><code>fast_mode</code></td>
                <td>Boolean</td>
                <td>If true, skips generating base64 visual previews to save memory and bandwidth. Default: <code>false</code>.</td>
              </tr>
            </tbody>
          </table>

          <h3>Response</h3>
          <pre className="code-block">
{`{
  "checkpoint": "aug",
  "quality": 95,
  "clean_score": 0.054, 
  "reencoded_score": 0.048, 
  "jpeg_kb": 124.5, // Only if fast_mode=false
  "warning": null,
  "clean_preview": "data:image/jpeg;base64,...", // Only if fast_mode=false
  "reencoded_preview": "data:image/jpeg;base64,...", // Only if fast_mode=false
  "ela_preview": "data:image/jpeg;base64,..." // Only if fast_mode=false
}`}
          </pre>
        </div>

        <div className="api-endpoint" style={{ marginTop: '48px' }}>
          <div className="endpoint-header">
            <span className="method">POST</span>
            <span className="path">/api/analyze-batch</span>
          </div>
          <p>Scores many images in one call. Mirrors the CLI tool's behavior and returns an array of scores.</p>
          
          <h3 style={{marginTop: 32}}>Parameters (FormData)</h3>
          <table className="api-table">
            <thead>
              <tr><th>Name</th><th>Type</th><th>Description</th></tr>
            </thead>
            <tbody>
              <tr>
                <td><code>images</code></td>
                <td>File[]</td>
                <td>Array of image files to analyze (Max 25MB each). Browsers don't expose paths, so the identifier is the filename.</td>
              </tr>
              <tr>
                <td><code>checkpoint</code></td>
                <td>String</td>
                <td>Model checkpoint to use. Options: <code>aug</code> (default), <code>baseline</code>.</td>
              </tr>
            </tbody>
          </table>

          <h3>Response</h3>
          <pre className="code-block">
{`{
  "results": [
    {
      "image_path": "DSC001.jpg",
      "pred": 0.054,
      "warning": null
    },
    {
      "image_path": "screenshot.png",
      "pred": 0.982,
      "warning": null
    }
  ]
}`}
          </pre>
        </div>
      </div>
    </div>
  )
}

/* ── Main App ───────────────────────────────────────────────── */


/* ── Custom SVG Stress Test Chart ───────────────────────────── */
function StressTestChart({ data }) {
  if (!data) return null;
  const qualities = [90, 70, 50, 30];
  const width = 500;
  const height = 250;
  const paddingX = 40;
  const paddingY = 40;
  const graphW = width - paddingX * 2;
  const graphH = height - paddingY * 2;
  
  const getX = (index) => paddingX + (index / (qualities.length - 1)) * graphW;
  const getY = (val) => height - paddingY - (val * graphH);

  const getPath = (points) => {
    if (!points || points.length === 0) return '';
    return points.map((val, i) => `${i === 0 ? 'M' : 'L'} ${getX(i)} ${getY(val)}`).join(' ');
  };

  return (
    <div style={{ width: '100%', overflowX: 'auto', background: 'rgba(0,0,0,0.2)', borderRadius: 'var(--radius-sm)', padding: '24px', marginTop: '24px', border: '1px solid var(--border)' }}>
      <svg viewBox={`0 0 ${width} ${height}`} style={{ width: '100%', minWidth: '400px', height: 'auto', display: 'block', overflow: 'visible' }}>
        {/* Grid lines */}
        {[0, 0.25, 0.5, 0.75, 1].map(tick => (
          <g key={tick}>
            <line x1={paddingX} y1={getY(tick)} x2={width - paddingX} y2={getY(tick)} stroke="var(--border)" strokeWidth="1" />
            <text x={paddingX - 10} y={getY(tick)} fill="var(--text-dim)" fontSize="10" textAnchor="end" alignmentBaseline="middle">
              {tick * 100}%
            </text>
          </g>
        ))}

        {/* X-axis labels */}
        {qualities.map((q, i) => (
          <text key={q} x={getX(i)} y={height - paddingY + 20} fill="var(--text-dim)" fontSize="10" textAnchor="middle">
            {q} Q
          </text>
        ))}

        {/* Baseline Model Path */}
        <motion.path 
          d={getPath(data.baseline)} 
          fill="none" 
          stroke="var(--text-dim)" 
          strokeWidth="2" 
          initial={{ pathLength: 0 }}
          animate={{ pathLength: 1 }}
          transition={{ duration: 1.5, ease: "easeInOut" }}
        />
        {data.baseline.map((val, i) => (
          <motion.circle 
            key={`b-${i}`} cx={getX(i)} cy={getY(val)} r="4" fill="var(--bg)" stroke="var(--text-dim)" strokeWidth="2" 
            initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ delay: 1.5 + i * 0.1 }}
          />
        ))}
        
        {/* Augmented Model Path */}
        <motion.path 
          d={getPath(data.aug)} 
          fill="none" 
          stroke="var(--brand-cyan)" 
          strokeWidth="3" 
          initial={{ pathLength: 0 }}
          animate={{ pathLength: 1 }}
          transition={{ duration: 1.5, ease: "easeInOut", delay: 0.2 }}
        />
        {data.aug.map((val, i) => (
          <motion.circle 
            key={`a-${i}`} cx={getX(i)} cy={getY(val)} r="5" fill="var(--brand-cyan)" stroke="var(--bg)" strokeWidth="2"
            initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ delay: 1.7 + i * 0.1 }}
          />
        ))}
      </svg>
      
      {/* Legend */}
      <div style={{ display: 'flex', justifyContent: 'center', gap: '24px', marginTop: '24px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div style={{ width: '12px', height: '3px', background: 'var(--brand-cyan)' }}></div>
          <span style={{ fontSize: '12px', color: 'var(--text-main)', fontWeight: 'bold' }}>Augmented Model (Ours)</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div style={{ width: '12px', height: '3px', borderTop: '2px dashed var(--text-dim)' }}></div>
          <span style={{ fontSize: '12px', color: 'var(--text-dim)' }}>Baseline Model</span>
        </div>
      </div>
    </div>
  );
}

/* ── Batch Analyzer ─────────────────────────────────────────── */

function BatchView() {
  const [files, setFiles] = useState([])
  const [checkpoint, setCheckpoint] = useState('aug')
  const [results, setResults] = useState(null)
  const [status, setStatus] = useState('idle')
  const [error, setError] = useState('')
  const inputRef = useRef(null)

  const addFiles = (fileList) => {
    const incoming = Array.from(fileList).filter((f) => /\.(jpe?g|png|webp|bmp)$/i.test(f.name))
    if (!incoming.length) return
    setFiles((prev) => {
      const seen = new Set()
      return [...prev, ...incoming]
        .filter((f) => {
          const key = `${f.name}:${f.size}`
          if (seen.has(key)) return false
          seen.add(key)
          return true
        })
        .slice(0, BATCH_LIMIT)
    })
    setResults(null)
    setStatus('idle')
    setError('')
  }

  const removeFile = (name) => {
    setFiles((prev) => prev.filter((f) => f.name !== name))
    setResults(null)
    setStatus('idle')
  }

  const clearAll = () => {
    setFiles([])
    setResults(null)
    setStatus('idle')
    setError('')
    if (inputRef.current) inputRef.current.value = ''
  }

  const runBatch = async () => {
    if (!files.length) return
    setStatus('loading')
    setError('')
    const form = new FormData()
    files.forEach((f) => form.append('images', f))
    form.append('checkpoint', checkpoint)
    try {
      const res = await fetch('/api/analyze-batch', { method: 'POST', body: form })
      const body = await res.json()
      if (!res.ok) throw new Error(body.detail || 'Batch analysis failed.')
      setResults(body.results)
      setStatus('done')
    } catch (caught) {
      setStatus('error')
      setError(caught.message)
    }
  }

  const downloadJson = () => {
    if (!results) return
    const preds = results.map((r) => ({ image_path: r.image_path, pred: r.pred }))
    const blob = new Blob([JSON.stringify(preds, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'preds.json'
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="tool-section-wrapper batch-view" style={{ paddingTop: '40px', paddingBottom: '80px' }}>
      <div className="tool-section-header">
        <h2>Batch Analyzer</h2>
        <p>The same scoring pipeline as <code>predict.py</code> — upload a folder of images at once and export the identical <code>preds.json</code> shape.</p>
      </div>

      <div className="batch-workspace">
        <div
          className={`drop-target${files.length ? ' active' : ''}`}
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => { e.preventDefault(); addFiles(e.dataTransfer.files) }}
          onClick={() => inputRef.current?.click()}
          style={{ cursor: 'pointer' }}
        >
          <input ref={inputRef} type="file" multiple accept="image/jpeg,image/png,image/webp,image/bmp" onChange={(e) => addFiles(e.target.files)} />
          <FileImage size={32} className="dim" />
          <div className="semibold">Click or drag images here</div>
          <div className="dim" style={{ fontSize: 12 }}>
            {files.length ? `${files.length} file${files.length === 1 ? '' : 's'} queued` : `JPG, PNG, WebP, BMP — up to ${BATCH_LIMIT} at a time`}
          </div>
        </div>

        {files.length > 0 && (
          <>
            <div className="batch-file-chips">
              {files.map((f) => (
                <span className="batch-chip" key={`${f.name}:${f.size}`}>
                  {f.name}
                  <button onClick={() => removeFile(f.name)} title="Remove"><X size={12} /></button>
                </span>
              ))}
            </div>

            <div className="batch-controls">
              <div className="segmented-control" style={{ width: 300, padding: '6px' }}>
                {Object.entries(CHECKPOINTS).map(([k, v]) => (
                  <button key={k} className={`seg-btn${checkpoint === k ? ' active' : ''}`} style={{ padding: '10px 16px', fontSize: '14px' }} onClick={() => setCheckpoint(k)}>{v}</button>
                ))}
              </div>
              <div style={{ display: 'flex', gap: 12 }}>
                <button className="btn-clear" style={{ width: 'auto', margin: 0 }} onClick={clearAll}>Clear</button>
                <button className="btn-primary" style={{ width: 'auto', padding: '10px 24px' }} onClick={runBatch} disabled={status === 'loading'}>
                  {status === 'loading' ? 'Analyzing…' : `Analyze ${files.length} Image${files.length === 1 ? '' : 's'}`}
                </button>
              </div>
            </div>
          </>
        )}

        {error && <div style={{ color: 'var(--red)', fontSize: 13, marginTop: 12 }}>{error}</div>}

        {results && status === 'done' && (
          <>
            <div className="batch-results-header">
              <span className="dim">{results.length} scored &middot; {results.filter((r) => r.pred == null).length} unreadable</span>
              <button className="btn-secondary" style={{ padding: '10px 20px', fontSize: 13 }} onClick={downloadJson}>Download preds.json</button>
            </div>
            <div className="batch-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: '24px', marginTop: '24px' }}>
              {results.map((r) => {
                const { verdict, tone } = classify(r.pred)
                const file = files.find(f => f.name === r.image_path)
                const imgUrl = file ? URL.createObjectURL(file) : null
                
                const scoreColor = tone === 'high' ? 'var(--red)' : tone === 'mid' ? 'var(--orange)' : 'var(--brand-cyan)'

                return (
                  <div className="card" key={r.image_path} style={{ overflow: 'hidden', padding: 0, display: 'flex', flexDirection: 'column' }}>
                    <div style={{ width: '100%', height: '160px', backgroundColor: 'var(--bg)', position: 'relative' }}>
                      {imgUrl ? (
                        <img src={imgUrl} alt={r.image_path} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                      ) : (
                        <div style={{ width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}><FileImage className="dim" /></div>
                      )}
                      {r.warning && (
                        <div style={{ position: 'absolute', top: 8, right: 8, background: 'rgba(0,0,0,0.7)', padding: '4px', borderRadius: '4px' }} title={r.warning}>
                          <AlertTriangle size={14} color="var(--orange)" />
                        </div>
                      )}
                    </div>
                    <div style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                        <div>
                          <div style={{ fontSize: '11px', textTransform: 'uppercase', color: 'var(--text-dim)', letterSpacing: '0.05em', marginBottom: '4px' }}>AI Signal</div>
                          <div style={{ fontSize: '24px', fontWeight: 'bold', color: scoreColor, lineHeight: 1 }}>
                            {r.pred == null ? '—' : `${(r.pred * 100).toFixed(1)}%`}
                          </div>
                        </div>
                        <span className={`batch-badge ${tone}`} style={{ margin: 0 }}>{verdict}</span>
                      </div>
                      <div className="mono dim" style={{ fontSize: '11px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }} title={r.image_path}>
                        {r.image_path}
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>
          </>
        )}
      </div>
    </div>
  )
}

/* ── Safety Guide ───────────────────────────────────────────── */

const VISUAL_TELLS = [
  { title: 'Look at the hands', body: 'Extra or missing fingers, or fingers that bend the wrong way.' },
  { title: 'Look at the writing', body: 'Signs, labels or captions that are blurry or spelled oddly.' },
  { title: 'Look at the background', body: 'Patterns that repeat, or objects that blend into each other.' },
  { title: 'Look at the skin', body: 'Skin that looks too smooth and even, almost like plastic.' },
]

const SCAM_PATTERNS = [
  { title: 'A new online "friend" or partner', body: 'Chats for weeks using an attractive photo, then asks for money.' },
  { title: 'A "celebrity" investment tip', body: 'A familiar face in a photo or video promotes a stock or crypto deal.' },
  { title: 'An urgent message from "family"', body: 'A photo or voice note asks for money right away and says not to tell anyone.' },
  { title: 'A too-good online listing', body: 'Product photos look perfect, but nothing arrives after you pay.' },
]

const REPORT_RESOURCES = [
  {
    title: 'ScamShield Helpline',
    body: 'Free, 24 hours a day. Call to check if something is a scam.',
    url: 'https://www.scamshield.gov.sg/',
    linkLabel: 'scamshield.gov.sg',
  },
  {
    title: 'Singapore Police Force',
    body: 'File a police report or read about current scams.',
    url: 'https://www.police.gov.sg/Advisories/Scams',
    linkLabel: 'police.gov.sg/Advisories/Scams',
  },
]

const LEARN_RESOURCES = [
  {
    title: 'CSA GoSafeOnline',
    body: 'Government tips for staying safe online.',
    url: 'https://www.csa.gov.sg/gosafeonline',
  },
  {
    title: 'How Singapore law protects you',
    body: 'Police can now freeze a scam victim’s bank transfers, and take down scam websites and fake profiles.',
    url: 'https://sso.agc.gov.sg/Act/PSA2025',
  },
  {
    title: "Singapore's rules on deepfakes",
    body: 'How the government is responding to AI-faked photos and videos.',
    url: 'https://www.imda.gov.sg/resources/blog/blog-articles/2024/07/3-things-sg-do-to-take-action-against-deepfakes',
  },
]

function LearnView() {
  return (
    <div className="tool-section-wrapper safety-guide">
      <div className="tool-section-header" style={{ maxWidth: 640 }}>
        <h2>Spot AI Images &amp; Scams</h2>
        <p>A simple guide to what to look for, and who to call if something feels wrong.</p>
      </div>

      <div className="card safety-hero-callout">
        <div className="safety-hero-icon"><Phone size={28} /></div>
        <div>
          <div className="safety-hero-label">Think you're being scammed?</div>
          <a href="tel:1799" className="safety-hero-number">Call 1799</a>
          <div className="safety-hero-sub">ScamShield Helpline &middot; free &middot; 24 hours a day</div>
        </div>
      </div>

      <section className="safety-section">
        <h3><Eye size={22} /> 4 signs a photo might be fake</h3>
        <div className="safety-list">
          {VISUAL_TELLS.map((t) => (
            <div className="safety-item" key={t.title}>
              <CheckCircle2 size={20} />
              <div>
                <div className="safety-item-title">{t.title}</div>
                <div className="safety-item-body">{t.body}</div>
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="safety-section">
        <h3><ShieldAlert size={22} /> Tricks that use fake photos</h3>
        <div className="safety-list">
          {SCAM_PATTERNS.map((t) => (
            <div className="safety-item" key={t.title}>
              <AlertTriangle size={20} />
              <div>
                <div className="safety-item-title">{t.title}</div>
                <div className="safety-item-body">{t.body}</div>
              </div>
            </div>
          ))}
        </div>
      </section>

      <div className="card safety-reassurance">
        <div className="safety-reassurance-title">If you're not sure, slow down</div>
        <ul>
          <li>Don't send money or personal details.</li>
          <li>Talk to someone you trust before you act.</li>
          <li>Call <a href="tel:1799">1799</a> to check — it's free.</li>
        </ul>
      </div>

      <section className="safety-more">
        <h4><Landmark size={16} /> Where to get help &amp; learn more</h4>

        <div className="safety-more-group">
          <div className="safety-more-label">Report a scam</div>
          {REPORT_RESOURCES.map((r) => (
            <a key={r.title} className="safety-more-row" href={r.url} target="_blank" rel="noopener noreferrer">
              <div>
                <div className="safety-more-row-title">{r.title}</div>
                <div className="safety-more-row-body">{r.body}</div>
              </div>
              <ExternalLink size={16} />
            </a>
          ))}
        </div>

        <div className="safety-more-group">
          <div className="safety-more-label">Learn more</div>
          {LEARN_RESOURCES.map((r) => (
            <a key={r.title} className="safety-more-row" href={r.url} target="_blank" rel="noopener noreferrer">
              <div>
                <div className="safety-more-row-title">{r.title}</div>
                <div className="safety-more-row-body">{r.body}</div>
              </div>
              <ExternalLink size={16} />
            </a>
          ))}
        </div>

        <p className="learn-disclaimer">
          This is general information for Singapore, not legal advice. Rules
          can change, so check the links above for the latest details.
        </p>
      </section>
    </div>
  )
}


/* ── Interactive Hero Demo ──────────────────────────────────── */
function HeroDemo() {
  return (
    <div className="hero-demo-container" style={{ perspective: '1200px', width: '100%', maxWidth: '800px', margin: '40px auto', height: '400px', position: 'relative', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      
      {/* Moving Trail Border Wrapper */}
      <motion.div
        animate={{ rotateY: [-5, 5, 10, -5], rotateX: [2, 5, 0, 2] }}
        transition={{ duration: 8, repeat: Infinity, ease: "easeInOut" }}
        style={{
          width: '100%', height: '100%', position: 'relative',
          transformStyle: 'preserve-3d',
          borderRadius: 'var(--radius-lg)',
          padding: '2px', // Space for the moving border
          overflow: 'hidden'
        }}
      >
        {/* The glowing conic gradient background that rotates */}
        <motion.div
          animate={{ rotate: [0, 360] }}
          transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
          style={{
            position: 'absolute', top: '-50%', left: '-50%', width: '200%', height: '200%',
            background: 'conic-gradient(from 0deg, transparent 70%, var(--brand-cyan) 80%, var(--red) 100%)',
            zIndex: 0
          }}
        />

        {/* The actual inner window */}
        <div
          className="demo-window"
          style={{
            width: '100%', height: '100%', background: '#09090b',
            borderRadius: 'calc(var(--radius-lg) - 2px)',
            boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.8), inset 0 1px 0 rgba(255,255,255,0.1)', overflow: 'hidden',
            display: 'flex', flexDirection: 'column', position: 'relative', zIndex: 1
          }}
        >
          <div style={{ height: '32px', borderBottom: '1px solid var(--border)', display: 'flex', alignItems: 'center', padding: '0 16px', gap: '8px', background: 'rgba(255,255,255,0.02)' }}>
            <div style={{ width: 10, height: 10, borderRadius: '50%', background: '#FE2C55' }} />
            <div style={{ width: 10, height: 10, borderRadius: '50%', background: '#F5A623' }} />
            <div style={{ width: 10, height: 10, borderRadius: '50%', background: '#25F4EE' }} />
          </div>
          
          <div style={{ display: 'flex', flex: 1, padding: '24px', gap: '24px' }}>
            {/* Fake Sidebar */}
            <div style={{ width: '30%', display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <div style={{ width: '100%', height: '120px', background: 'rgba(255,255,255,0.03)', borderRadius: 'var(--radius-md)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                 <ImageIcon color="var(--text-dim)" size={32} />
              </div>
              
              <motion.button 
                className="btn-primary" 
                style={{ padding: '12px', fontSize: '14px', position: 'relative', overflow: 'hidden' }}
                animate={{ scale: [1, 1, 0.92, 1, 1], background: ['var(--brand-cyan)', 'var(--brand-cyan)', '#1BA5A1', 'var(--brand-cyan)', 'var(--brand-cyan)'] }}
                transition={{ duration: 4, repeat: Infinity, times: [0, 0.4, 0.45, 0.5, 1], ease: "easeInOut" }}
              >
                Analyze Image
                <motion.div 
                  style={{ position: 'absolute', top: '50%', left: '50%', width: '100%', height: '100%', background: 'rgba(255,255,255,0.4)', borderRadius: '50%', x: '-50%', y: '-50%' }}
                  animate={{ scale: [0, 0, 2, 0, 0], opacity: [0, 0, 1, 0, 0] }}
                  transition={{ duration: 4, repeat: Infinity, times: [0, 0.45, 0.46, 0.6, 1] }}
                />
              </motion.button>
              <div style={{ width: '100%', height: '8px', background: 'rgba(255,255,255,0.03)', borderRadius: '4px' }} />
              <div style={{ width: '80%', height: '8px', background: 'rgba(255,255,255,0.03)', borderRadius: '4px' }} />
            </div>

            {/* Fake Dashboard */}
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <motion.div 
                style={{ width: '100%', padding: '20px', background: 'rgba(254, 44, 85, 0.1)', borderRadius: 'var(--radius-md)', border: '1px solid rgba(254, 44, 85, 0.2)' }}
                animate={{ opacity: [0, 0, 1, 1, 0], y: [10, 10, 0, 0, 10] }}
                transition={{ duration: 4, repeat: Infinity, times: [0, 0.55, 0.65, 0.9, 1], ease: "easeOut" }}
              >
                <div style={{ color: '#FE2C55', fontWeight: 'bold', fontSize: '14px', marginBottom: '8px' }}>Likely AI-Generated</div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ fontSize: '12px', color: 'var(--text-dim)' }}>AI Signal Score</span>
                  <span style={{ fontSize: '12px', color: '#FE2C55', fontWeight: 'bold' }}>92.4%</span>
                </div>
              </motion.div>
              
              <div style={{ display: 'flex', gap: '16px', flex: 1 }}>
                <div style={{ flex: 1, background: 'rgba(255,255,255,0.02)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border)' }} />
                <div style={{ flex: 1, background: 'rgba(255,255,255,0.02)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border)' }} />
              </div>
            </div>
          </div>

          {/* Animated Cursor */}
          <motion.div
            style={{
              position: 'absolute',
              zIndex: 100,
              width: '28px',
              height: '28px',
              pointerEvents: 'none',
              filter: 'drop-shadow(0 4px 6px rgba(0,0,0,0.5))'
            }}
            animate={{ 
              x: ['400px', '130px', '130px', '400px', '400px'],
              y: ['300px', '180px', '180px', '300px', '300px'],
              scale: [1, 1, 0.7, 1, 1]
            }}
            transition={{ duration: 4, repeat: Infinity, times: [0, 0.4, 0.45, 0.5, 1], ease: "easeInOut" }}
          >
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="m3 3 7.07 16.97 2.51-7.39 7.39-2.51L3 3z" fill="black" />
            </svg>
            
            {/* Ripple Click Effect */}
            <motion.div
              style={{ position: 'absolute', top: '4px', left: '4px', width: '20px', height: '20px', border: '2px solid white', borderRadius: '50%' }}
              animate={{ scale: [0, 0, 2.5, 0, 0], opacity: [0, 0, 0.8, 0, 0] }}
              transition={{ duration: 4, repeat: Infinity, times: [0, 0.44, 0.45, 0.6, 1] }}
            />
          </motion.div>
        </div>
      </motion.div>
    </div>
  )
}

export default function App() {
  const [theme, setTheme] = useState(() =>
    localStorage.getItem('aigc-theme') ||
    (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'),
  )
  const [view, setView] = useState('home')
  const [file, setFile] = useState(null)
  const [checkpoint, setCheckpoint] = useState('aug')
  const [quality, setQuality] = useState(95)
  const [result, setResult] = useState(null)
  const [status, setStatus] = useState('idle')
  const [error, setError] = useState('')
  const [deepScan, setDeepScan] = useState(false)
  const [modalImage, setModalImage] = useState(null) // { type: 'image' | 'heatmap', src: string }
  const [shareCopied, setShareCopied] = useState(false)
  const [shareModal, setShareModal] = useState(null)
  const [stressData, setStressData] = useState(null)
  const [stressStatus, setStressStatus] = useState('idle')
  const [stressProgress, setStressProgress] = useState(0) // idle, loading, done, error
  const inputRef = useRef(null)

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    localStorage.setItem('aigc-theme', theme)
  }, [theme])

  const analyze = useCallback(
    async (nextFile = file, nextModel = checkpoint, nextQuality = quality) => {
      if (!nextFile) return
      setStatus('loading')
      setError('')
      const form = new FormData()
      form.append('image', nextFile)
      form.append('checkpoint', nextModel)
      form.append('quality', `${nextQuality}`)
      try {
        const res = await fetch('/api/analyze', {
          method: 'POST',
          body: form,
        })
        const body = await res.json()
        if (!res.ok) throw new Error(body.detail || 'Analysis failed.')
        setResult(body)
        setStatus('done')
      } catch (caught) {
        setStatus('error')
        setError(caught.message)
      }
    },
    [file, checkpoint, quality],
  )

  const chooseFile = (f) => {
    if (!f) return
    if (f.size > 25 * 1024 * 1024) {
      setError('File is too large. The limit is 25MB.')
      setStatus('error')
      return
    }
    setFile(f)
    setResult(null)
    setStressData(null)
    setStressStatus('idle')
    analyze(f)
  }

  
  const runStressTest = async () => {
    if (!file) return;
    setStressStatus('loading');
    setStressData(null);
    
    const qualities = [90, 70, 50, 30];
    const results = { aug: [], baseline: [] };
    
    try {
      let step = 1;
      for (const q of qualities) {
        setStressProgress(step);
        step++;
        const fdAug = new FormData();
        fdAug.append('image', file);
        fdAug.append('checkpoint', 'aug');
        fdAug.append('quality', q.toString());
        fdAug.append('fast_mode', 'true');
        const resAug = await fetch('/api/analyze', { method: 'POST', body: fdAug });
        const dataAug = await resAug.json();
        if (!resAug.ok) throw new Error(dataAug.detail || 'Aug model failed');
        results.aug.push(dataAug.reencoded_score !== undefined ? dataAug.reencoded_score : dataAug.clean_score);
        
        const fdBase = new FormData();
        fdBase.append('image', file);
        fdBase.append('checkpoint', 'baseline');
        fdBase.append('quality', q.toString());
        fdBase.append('fast_mode', 'true');
        const resBase = await fetch('/api/analyze', { method: 'POST', body: fdBase });
        const dataBase = await resBase.json();
        if (!resBase.ok) throw new Error(dataBase.detail || 'Baseline model failed');
        results.baseline.push(dataBase.reencoded_score !== undefined ? dataBase.reencoded_score : dataBase.clean_score);
      }
      console.log('Stress Test Results:', results);
      setStressData(results);
      setStressStatus('done');
    } catch (err) {
      console.error('Stress Test Error:', err);
      alert('Stress Test Error: ' + err.message);
      setStressStatus('error');
    }
  }

  const clearFile = () => {
    setFile(null)
    setResult(null)
    setStressData(null)
    setStressStatus('idle')
    setStatus('idle')
    setError('')
    if (inputRef.current) inputRef.current.value = ''
  }

  const handleQualityChange = (e) => {
    setQuality(Number(e.target.value))
  }
  const handleQualityCommit = () => {
    if (file) analyze()
  }

  return (
    <div className="app-container">
      <AnimatePresence>
        {shareModal && (
          <motion.div
            className="modal-overlay"
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            onClick={(e) => {
              if (e.target === e.currentTarget) {
                URL.revokeObjectURL(shareModal.url);
                setShareModal(null);
              }
            }}
          >
            <motion.div
              className="modal-content"
              initial={{ scale: 0.95, opacity: 0, y: 20 }}
              animate={{ scale: 1, opacity: 1, y: 0 }}
              exit={{ scale: 0.95, opacity: 0, y: 20 }}
              style={{ maxWidth: 400, background: 'var(--panel-bg)', borderRadius: 'var(--radius-lg)', padding: '24px', display: 'flex', flexDirection: 'column', alignItems: 'center', border: '1px solid var(--border)', zIndex: 9999 }}
            >
              <h3 style={{ margin: '0 0 16px 0' }}>Share Result</h3>
              <img src={shareModal.url} alt="Share Preview" style={{ width: '100%', borderRadius: 'var(--radius-md)', border: '1px solid var(--border)', marginBottom: '24px' }} />
              
              <div style={{ display: 'flex', gap: '12px', width: '100%' }}>
                <button 
                  className="btn-secondary" 
                  style={{ flex: 1, padding: '12px' }}
                  onClick={() => {
                    URL.revokeObjectURL(shareModal.url);
                    setShareModal(null);
                  }}
                >
                  Cancel
                </button>
                <button 
                  className="btn-primary" 
                  style={{ flex: 1, padding: '12px', display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '8px' }}
                  onClick={async () => {
                    if (navigator.canShare && navigator.canShare(shareModal.shareData)) {
                      try {
                        await navigator.share(shareModal.shareData);
                      } catch (err) {
                        console.error('Share failed:', err);
                      }
                    } else {
                      const a = document.createElement('a');
                      a.href = shareModal.url;
                      a.download = 'imagesignal-verdict.png';
                      a.click();
                    }
                    URL.revokeObjectURL(shareModal.url);
                    setShareModal(null);
                  }}
                >
                  <Share2 size={16} /> Share Image
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {modalImage && (
        <div className="modal-overlay" onClick={() => setModalImage(null)}>
          <div className="modal-content" onClick={e => e.stopPropagation()}>
            <button className="modal-close" onClick={() => setModalImage(null)}><X size={32} strokeWidth={1.5} /></button>
            {modalImage.type === 'heatmap' ? (
              <ElaHeatmap src={modalImage.src} fullscreen={true} />
            ) : (
              <img src={modalImage.src} alt="Fullscreen View" />
            )}
          </div>
        </div>
      )}

      <AnimatedBackground />
      <Header theme={theme} onToggleTheme={() => setTheme(t => t === 'dark' ? 'light' : 'dark')} currentView={view} setView={setView} />

      <div className="main-content-area" style={{ flex: 1, display: 'flex', flexDirection: 'column', position: 'relative' }}>
        <AnimatePresence mode="wait">
          {view === 'brief' && (
            <motion.iframe
              key="brief"
              src="/project-brief.html"
              className="brief-frame"
              title="Project Brief"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3, ease: 'easeInOut' }}
              style={{ width: '100%', border: 'none', flex: 1 }}
            />
          )}

          
          {view === 'transition' && (
            <motion.div
              key="transition"
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, y: -50, scale: 1.1 }}
              transition={{ duration: 0.4, ease: 'easeInOut' }}
              style={{ flex: 1, display: 'flex' }}
            >
              <TransitionBot onComplete={() => setView('tool')} />
            </motion.div>
          )}
          {view === 'home' && (
            <motion.div
              key="home"
              className="home-view"
              initial={{ opacity: 0, scale: 0.98, filter: 'blur(5px)' }}
              animate={{ opacity: 1, scale: 1, filter: 'blur(0px)' }}
              exit={{ opacity: 0, scale: 0.98, filter: 'blur(5px)' }}
              transition={{ duration: 0.4, ease: 'easeInOut' }}
              style={{ flex: 1, display: 'flex', flexDirection: 'column' }}
            >
              <div className="hero-section" style={{ position: 'relative', width: '100%', minHeight: '40vh', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
                <h1 className="hero-title" style={{ position: 'relative', zIndex: 20 }}>Detect AI-Generated Images with <span>Confidence</span></h1>
                <p className="hero-subtitle" style={{ position: 'relative', zIndex: 20 }}>
                  Instantly analyze images to determine if they were synthetically generated or human-made, 
                  using state-of-the-art model checkpointing and compression forensics.
                </p>
                <div className="hero-actions" style={{ position: 'relative', zIndex: 20 }}>
                  <button 
                    className="btn-large" 
                    onClick={() => setView('transition')}
                  >
                    Start Analysis
                  </button>
                  <button 
                    className="btn-secondary"
                    onClick={() => setView('brief')}
                  >
                    Read the Brief
                  </button>
                </div>
              </div>

              <div className="video-section" style={{ overflow: 'hidden' }}>
                <HeroDemo />
              </div>
            </motion.div>
          )}

          {view === 'api' && (
            <motion.div
              key="api"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.4, ease: 'easeInOut' }}
              style={{ flex: 1, overflowY: 'auto' }}
            >
              <ApiDocs />
            </motion.div>
          )}

          {view === 'batch' && (
            <motion.div
              key="batch"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.4, ease: 'easeInOut' }}
              style={{ flex: 1, overflowY: 'auto' }}
            >
              <BatchView />
            </motion.div>
          )}

          {view === 'learn' && (
            <motion.div
              key="learn"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.4, ease: 'easeInOut' }}
              style={{ flex: 1, overflowY: 'auto' }}
            >
              <LearnView />
            </motion.div>
          )}

          {view === 'tool' && (
            <motion.div
              key="tool"
              className="tool-view"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.4, ease: 'easeInOut' }}
              style={{ flex: 1 }}
            >
              <div className="tool-section-wrapper" id="analyzer-tool" style={{ paddingTop: '40px', paddingBottom: '80px', borderTop: 'none' }}>
                <div className="tool-section-header">
                  <h2>Interactive Analyzer</h2>
                  <p>Experience the detection engine live. Upload an image to test.</p>
                </div>
                <div className="workspace">
            {/* ── Left Column: Input Panel ───────────────────────────── */}
            <aside className="panel">
              
              <div 
                className={`drop-target${file ? ' active' : ''}`}
                onDragOver={e => e.preventDefault()}
                onDrop={e => { e.preventDefault(); chooseFile(e.dataTransfer.files[0]) }}
                onClick={() => { if (!file) inputRef.current?.click() }}
                style={{ cursor: file ? 'default' : 'pointer' }}
              >
                <input ref={inputRef} type="file" accept="image/jpeg,image/png,image/webp,image/bmp" onChange={e => chooseFile(e.target.files[0])} />
                
                {!file ? (
                  <>
                    <FileImage size={32} className="dim" />
                    <div className="semibold">Click or drag an image here</div>
                    <div className="dim" style={{fontSize: 12}}>JPG, PNG, WebP up to 25MB</div>
                  </>
                ) : (
                  <div className="file-info">
                    <div className="val">{file.name}</div>
                    <div className="dim" style={{fontSize: 12}}>{(file.size / 1024 / 1024).toFixed(2)} MB</div>
                  </div>
                )}
              </div>

              {file && <button className="btn-clear" onClick={clearFile}>Remove Image</button>}

              <hr style={{ border: 0, borderTop: '1px solid var(--border)', margin: '32px 0' }} />

              <div className="control-block">
                <div className="control-label">
                  <span>AI Model</span>
                </div>
                <div className="segmented-control">
                  {Object.entries(CHECKPOINTS).map(([k, v]) => (
                    <button 
                      key={k} 
                      className={`seg-btn${checkpoint === k ? ' active' : ''}`} 
                      onClick={() => { setCheckpoint(k); if(file) analyze(file, k, quality); }}
                    >
                      {v}
                    </button>
                  ))}
                </div>
              </div>

              <div className="control-block">
                <div className="control-label">
                  <span>JPEG Quality Test</span>
                  <output>Q{quality}</output>
                </div>
                <input 
                  type="range" 
                  className="range-slider"
                  min="30" max="95" 
                  value={quality} 
                  onChange={handleQualityChange}
                  onMouseUp={handleQualityCommit}
                  onTouchEnd={handleQualityCommit}
                />
                <div style={{display:'flex', justifyContent:'space-between', fontSize: 11, color: 'var(--text-dim)', marginTop: 8}}>
                  <span>Q30 (High compression)</span>
                  <span>Q95 (Original)</span>
                </div>
              </div>

              <button 
                className="btn-primary" 
                onClick={() => analyze()} 
                disabled={!file || status === 'loading' || stressStatus === 'loading'}
              >
                {status === 'loading' ? 'Analyzing...' : 'Analyze Image'}
              </button>
              {file && status === 'done' && (
                <button 
                  className="btn-secondary" 
                  style={{ marginTop: 12, width: '100%', fontSize: '14px', padding: '12px' }}
                  onClick={runStressTest}
                  disabled={stressStatus === 'loading'}
                >
                  {stressStatus === 'loading' ? `Running Test (${stressProgress}/4)...` : 'Run Robustness Stress Test'}
                </button>
              )}
              {error && <div style={{color:'var(--red)', fontSize: 13, marginTop: 16, textAlign: 'center'}}>{error}</div>}
            </aside>

            {/* ── Right Column: Dashboard ────────────────────────────── */}
            <main className="dashboard-grid">
              {!result && status !== 'loading' && (
                <div className="empty-state">
                  <ImageIcon size={48} strokeWidth={1} />
                  <h2>Upload an image to start</h2>
                  <p>ImageSignal will detect if the image was generated by AI.</p>
                </div>
              )}

              {status === 'loading' && (
                <div className="empty-state">
                  <div className="spinner"></div>
                  <h2>Analyzing image...</h2>
                  <p>Looking for synthetic patterns and compression artifacts.</p>
                </div>
              )}

              {result && status === 'done' && (() => {
                const { verdict, detail, tone, isAI } = classify(result.reencoded_score)
                const scoreVal = result.reencoded_score != null ? (result.reencoded_score * 100).toFixed(1) : 'ERR'
                const cleanVal = result.clean_score != null ? (result.clean_score * 100).toFixed(1) : 'ERR'
                const delta = result.clean_score != null && result.reencoded_score != null ? ((result.reencoded_score - result.clean_score) * 100).toFixed(1) : '0.0'
                const position = result.reencoded_score != null ? result.reencoded_score * 100 : 50

                const handleShare = async () => {
                  setShareCopied(true);
                  try {
                    const blob = await generateShareCard(result.clean_preview, verdict, scoreVal, tone, isAI);
                    const file = new File([blob], 'imagesignal-verdict.png', { type: 'image/png' });
                    const url = URL.createObjectURL(blob);
                    
                    const shareData = {
                      title: 'ImageSignal Analysis',
                      text: `ImageSignal detected this image is ${scoreVal}% likely to be ${isAI ? 'AI-Generated' : 'Authentic'}!`,
                      files: [file]
                    };
                    
                    setShareModal({ url, shareData, blob });
                  } catch (err) {
                    console.error('Failed to generate share image', err);
                  } finally {
                    setShareCopied(false);
                  }
                };

                return (
                  <motion.div 
                    initial="hidden"
                    animate="visible"
                    variants={{
                      hidden: { opacity: 0 },
                      visible: {
                        opacity: 1,
                        transition: { staggerChildren: 0.1 }
                      }
                    }}
                    style={{ display: 'contents' }}
                  >
                    {result.warning && (
                      <motion.div variants={{ hidden: { opacity: 0, y: 10 }, visible: { opacity: 1, y: 0 } }} className="warning-banner">
                        <AlertTriangle size={20} /> 
                        <span>{result.warning}</span>
                      </motion.div>
                    )}
                    
                    <motion.div variants={{ hidden: { opacity: 0, scale: 0.95 }, visible: { opacity: 1, scale: 1 } }} className={`card verdict-banner ${tone}`} style={{ position: 'relative' }}>
                      <div className={`verdict-icon ${tone}`}>
                        {isAI ? <Bot size={28} /> : <User size={28} />}
                      </div>
                      <div className="verdict-text">
                        <div className="verdict-eyebrow">Detection Result</div>
                        <h1>{verdict}</h1>
                        <p>{detail}</p>
                      </div>
                      <button 
                        onClick={handleShare} 
                        className="share-btn"
                        title="Share Result"
                      >
                        {shareCopied ? <CheckCircle2 size={16} /> : <Share2 size={16} />}
                        <span>{shareCopied ? 'Shared!' : 'Share'}</span>
                      </button>
                    </motion.div>

                    <motion.div variants={{ hidden: { opacity: 0, y: 20 }, visible: { opacity: 1, y: 0 } }} className="card data-matrix">
                      <div className="data-cell">
                        <div className="data-label uppercase">AI Signal Score</div>
                        <div className="data-value" style={{color: `var(--${tone === 'high' ? 'red' : tone === 'mid' ? 'amber' : 'green'})`}}>{scoreVal}%</div>
                        <div className="data-sub">Primary Indicator</div>
                      </div>
                      <div className="data-cell">
                        <div className="data-label uppercase">Base Model Confidence</div>
                        <div className="data-value">{cleanVal}%</div>
                        <div className="data-sub">Pre-compression</div>
                      </div>
                      <div className="data-cell">
                        <div className="data-label uppercase">Compression Delta</div>
                        <div className="data-value">{delta > 0 ? '+' : ''}{delta}%</div>
                        <div className="data-sub">Sensitivity</div>
                      </div>
                      <div className="data-cell">
                        <div className="data-label uppercase">File Size</div>
                        <div className="data-value">{result.jpeg_kb} KB</div>
                        <div className="data-sub">At Q{result.quality}</div>
                      </div>
                    </motion.div>

                    <motion.div variants={{ hidden: { opacity: 0, y: 20 }, visible: { opacity: 1, y: 0 } }} className="card spectrum-card">
                      <div className="verdict-eyebrow">Confidence Spectrum</div>
                      <div className="spectrum-bar">
                        <div className="spectrum-marker" style={{ '--pos': `${position}%` }}>
                          {isAI ? <Bot size={14} /> : <User size={14} />}
                        </div>
                      </div>
                      <div className="spectrum-labels">
                        <span style={{color: 'var(--green)'}}>More Human</span>
                        <span style={{color: 'var(--red)'}}>More AI</span>
                      </div>
                    </motion.div>

                    <motion.div variants={{ hidden: { opacity: 0, y: 20 }, visible: { opacity: 1, y: 0 } }} className="proofs-grid">
                      <div className="card proof-panel">
                        <div className="proof-header">
                          <span>Original Image</span>
                        </div>
                        <div className="proof-image">
                          {!result.clean_preview ? (
                             <FileImage size={48} className="dim" />
                          ) : (
                             <>
                               <button className="expand-btn" onClick={() => setModalImage({ type: 'image', src: result.clean_preview })}><Maximize size={20} /></button>
                               <img src={result.clean_preview} alt="Original" />
                             </>
                          )}
                        </div>
                      </div>
                      <div className="card proof-panel">
                        <div className="proof-header">
                          <span>Compressed Result</span>
                          <span className="dim">Q{result.quality}</span>
                        </div>
                        <div className="proof-image">
                          {result.reencoded_preview ? (
                            <>
                              <button className="expand-btn" onClick={() => setModalImage({ type: 'image', src: result.reencoded_preview })}><Maximize size={20} /></button>
                              <img src={result.reencoded_preview} alt="Compressed" />
                            </>
                          ) : <FileImage size={48} className="dim" />}
                        </div>
                      </div>
                    </motion.div>

                    {stressData && (
                      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
                        <div className="card" style={{ padding: '24px', marginTop: '24px' }}>
                          <h3 style={{ margin: '0 0 8px 0', fontSize: '18px' }}>Robustness Degradation Curve</h3>
                          <p style={{ margin: '0 0 16px 0', fontSize: '13px', color: 'var(--text-dim)' }}>
                            Simulating the effect of heavy social media compression (90% to 30% JPEG quality) on the model's confidence.
                          </p>
                          <StressTestChart data={stressData} />
                        </div>
                      </motion.div>
                    )}
                  </motion.div>
                )
              })()}
            </main>
          </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  )
}
