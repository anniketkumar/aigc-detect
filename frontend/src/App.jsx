import { useCallback, useEffect, useRef, useState } from 'react'
import {
  AlertTriangle,
  Bot,
  FileImage,
  ImageIcon,
  User,
  PlayCircle,
} from 'lucide-react'
import { motion } from 'framer-motion'
import './index.css'

/* ── Constants ──────────────────────────────────────────────── */

const CHECKPOINTS = {
  aug: 'Augmented',
  baseline: 'Baseline',
}

/* ── Helpers ────────────────────────────────────────────────── */

function classify(score) {
  if (score == null) return { verdict: 'Unable to analyze', detail: 'This image could not be processed. Please try another.', tone: 'neutral', isAI: false }
  if (score >= 0.7) return { verdict: 'Likely AI-Generated', detail: 'We found strong indicators that this image was synthetically generated.', tone: 'high', isAI: true }
  if (score >= 0.4) return { verdict: 'Inconclusive', detail: 'The image has mixed signals. It may be heavily edited or AI-generated.', tone: 'mid', isAI: false }
  return { verdict: 'Likely Authentic', detail: 'We found very few synthetic indicators. This appears to be a human-made image.', tone: 'low', isAI: false }
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

/* ── Header ─────────────────────────────────────────────────── */

function Header({ theme, onToggleTheme, currentView, setView }) {
  return (
    <div className="top-bar">
      <div className="brand">ImageSignal</div>
      <div className="nav-links">
        <button className={`nav-btn ${currentView === 'home' ? 'active' : ''}`} onClick={() => setView('home')}>Tool</button>
        <button className={`nav-btn ${currentView === 'brief' ? 'active' : ''}`} onClick={() => setView('brief')}>Project Brief</button>
      </div>
      <button className="theme-btn" onClick={onToggleTheme} title="Toggle Dark Mode">
        {theme === 'dark' ? 'Light Mode' : 'Dark Mode'}
      </button>
    </div>
  )
}

/* ── Main App ───────────────────────────────────────────────── */

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
    analyze(f)
  }

  const clearFile = () => {
    setFile(null)
    setResult(null)
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
      <AnimatedBackground />
      <Header theme={theme} onToggleTheme={() => setTheme(t => t === 'dark' ? 'light' : 'dark')} currentView={view} setView={setView} />

      {view === 'brief' ? (
        <iframe src="/project-brief.html" className="brief-frame" title="Project Brief" />
      ) : (
        <div className="home-view">
          <motion.div 
            className="hero-section"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, ease: "easeOut" }}
          >
            <h1 className="hero-title">Detect AI-Generated Images with <span>Confidence</span></h1>
            <p className="hero-subtitle">
              Instantly analyze images to determine if they were synthetically generated or human-made, 
              using state-of-the-art model checkpointing and compression forensics.
            </p>
            <div className="hero-actions">
              <button 
                className="btn-large" 
                onClick={() => document.getElementById('analyzer-tool').scrollIntoView({ behavior: 'smooth' })}
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
          </motion.div>

          <motion.div 
            className="video-section"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.2, ease: "easeOut" }}
          >
            <div className="video-mockup">
              <div className="mockup-header">
                <div className="mockup-dots">
                  <div className="mockup-dot r"></div>
                  <div className="mockup-dot y"></div>
                  <div className="mockup-dot g"></div>
                </div>
              </div>
              <div className="mockup-body" onClick={() => alert("Video player would open here!")}>
                <PlayCircle size={64} className="play-icon" strokeWidth={1.5} />
                <span className="semibold">Watch Feature Overview</span>
              </div>
            </div>
          </motion.div>

          <div className="tool-section-wrapper" id="analyzer-tool">
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
                disabled={!file || status === 'loading'}
              >
                {status === 'loading' ? 'Analyzing...' : 'Analyze Image'}
              </button>
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
                    
                    <motion.div variants={{ hidden: { opacity: 0, scale: 0.95 }, visible: { opacity: 1, scale: 1 } }} className={`card verdict-banner ${tone}`}>
                      <div className={`verdict-icon ${tone}`}>
                        {isAI ? <Bot size={28} /> : <User size={28} />}
                      </div>
                      <div className="verdict-text">
                        <div className="verdict-eyebrow">Detection Result</div>
                        <h1>{verdict}</h1>
                        <p>{detail}</p>
                      </div>
                    </motion.div>

                    <motion.div variants={{ hidden: { opacity: 0, y: 20 }, visible: { opacity: 1, y: 0 } }} className="card data-matrix">
                      <div className="data-cell">
                        <div className="data-label uppercase">AI Signal Score</div>
                        <div className="data-value" style={{color: `var(--${tone === 'high' ? 'red' : tone === 'mid' ? 'amber' : 'green'})`}}>{scoreVal}%</div>
                        <div className="data-sub">After JPEG compression</div>
                      </div>
                      <div className="data-cell">
                        <div className="data-label uppercase">Original Score</div>
                        <div className="data-value">{cleanVal}%</div>
                        <div className="data-sub">Clean decode</div>
                      </div>
                      <div className="data-cell">
                        <div className="data-label uppercase">Score Change</div>
                        <div className="data-value">{delta > 0 ? '+' : ''}{delta} pts</div>
                        <div className="data-sub">Shift upon compression</div>
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
                          {result.clean_preview ? <img src={result.clean_preview} alt="Original" /> : <FileImage size={48} className="dim" />}
                        </div>
                      </div>
                      <div className="card proof-panel">
                        <div className="proof-header">
                          <span>Compressed Result</span>
                          <span className="dim">Q{result.quality}</span>
                        </div>
                        <div className="proof-image">
                          {result.reencoded_preview ? <img src={result.reencoded_preview} alt="Compressed" /> : <FileImage size={48} className="dim" />}
                        </div>
                      </div>
                    </motion.div>
                  </motion.div>
                )
              })()}
            </main>
          </div>
          </div>
        </div>
      )}
    </div>
  )
}

