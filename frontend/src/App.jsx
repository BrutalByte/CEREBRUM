import { useState, useEffect } from 'react'
import { useTelemetry } from './hooks/useTelemetry'
import { Brain3D }       from './components/Brain3D'
import { QueryPanel }    from './components/QueryPanel'
import { ChemicalPanel } from './components/ChemicalPanel'
import { EventFeed }     from './components/EventFeed'
import { LoopPanel }     from './components/LoopPanel'
import { ProvenancePanel } from './components/ProvenancePanel'

import { GoalPanel }     from './components/GoalPanel'

export default function App() {
  const { events, connected } = useTelemetry()
  const [view, setView]                 = useState(() => localStorage.getItem('cerebrum_view') || 'graph')
  const [showQuery, setShowQuery]       = useState(false)
  const [showChemical, setShowChemical] = useState(false)
  const [showFeed, setShowFeed]         = useState(false)
  const [showLabels, setShowLabels]     = useState(() => localStorage.getItem('cerebrum_show_labels') === 'true')
  const [showAuto, setShowAuto]         = useState(false)
  const [showGoals, setShowGoals]       = useState(false)
  const [queryPaths, setQueryPaths]     = useState([])

  // Persistence
  useEffect(() => { localStorage.setItem('cerebrum_view', view) }, [view])
  useEffect(() => { localStorage.setItem('cerebrum_show_labels', showLabels) }, [showLabels])

  return (
    <div style={s.root}>
      {/* ── Main View Area ── */}
      <div style={s.mainArea}>
        {view === 'graph' ? (
          <Brain3D events={events} connected={connected} queryPaths={queryPaths} showLabels={showLabels} />
        ) : (
          <iframe
            src="/ui/dashboard.html"
            style={{ width: '100%', height: '100%', border: 'none' }}
            title="Dashboard"
          />
        )}
      </div>

      {/* ── Top bar ── */}
      <header style={s.topbar}>
        <span style={s.logo}>CEREBRUM</span>
        <span style={s.sep} />
        <span style={s.sub}>Neural Knowledge Graph</span>
        <span style={{ flex: 1 }} />
        
        <div style={s.navGroup}>
          <NavBtn active={view === 'graph'} onClick={() => setView('graph')}>3D Graph</NavBtn>
          <NavBtn active={view === 'dashboard'} onClick={() => setView('dashboard')}>Dashboard</NavBtn>
        </div>

        <span style={s.sep} />

        <NavBtn active={showFeed}     onClick={() => setShowFeed(v => !v)}>Events</NavBtn>
        <NavBtn active={showChemical} onClick={() => setShowChemical(v => !v)}>Metabolics</NavBtn>
        <NavBtn 
          active={showLabels}   
          onClick={() => {
            if (!showLabels) {
              const ok = window.confirm("Enabling all labels on a large graph can significantly slow down the 3D view. Proceed?")
              if (!ok) return
            }
            setShowLabels(v => !v)
          }}
        >
          Labels
        </NavBtn>
        <NavBtn active={showQuery}    onClick={() => setShowQuery(v => !v)}>Query</NavBtn>
        <NavBtn active={showGoals}    onClick={() => setShowGoals(v => !v)}>Goals</NavBtn>
        <NavBtn active={showAuto}     onClick={() => setShowAuto(v => !v)}>Autonomous</NavBtn>
        <span style={{ ...s.dot, background: connected ? '#3fb950' : '#f85149' }} />
        <span style={s.wsLabel}>{connected ? 'LIVE' : 'OFFLINE'}</span>
      </header>

      {/* ── Right sidebar ── */}
      {(showQuery || showChemical || showAuto || showGoals) && (
        <aside style={s.sidebar}>
          {showQuery && (
            <Panel title="Query">
              <QueryPanel onAnswers={setQueryPaths} />
            </Panel>
          )}
          {showChemical && (
            <Panel title="Chemical Modulator">
              <ChemicalPanel events={events} />
            </Panel>
          )}
          {showGoals && (
            <Panel title="System Goals">
              <GoalPanel />
            </Panel>
          )}
          {showAuto && (
            <>
              <Panel title="Loop Controller">
                <LoopPanel />
              </Panel>
              <Panel title="Provenance Ledger">
                <ProvenancePanel />
              </Panel>
            </>
          )}
        </aside>
      )}

      {/* ── Bottom event feed ── */}
      {showFeed && (
        <div style={s.feedBar}>
          <EventFeed events={events} connected={connected} />
        </div>
      )}
    </div>
  )
}

function NavBtn({ children, active, onClick }) {
  return (
    <button onClick={onClick} style={{ ...s.navBtn, ...(active ? s.navBtnActive : {}) }}>
      {children}
    </button>
  )
}

function Panel({ title, children }) {
  return (
    <div style={s.panel}>
      <div style={s.panelTitle}>{title}</div>
      {children}
    </div>
  )
}

const s = {
  root: {
    position: 'fixed',
    inset: 0,
    background: '#000',
    color: '#c9d1d9',
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
    fontSize: 13,
    overflow: 'hidden',
  },
  mainArea: {
    position: 'absolute',
    inset: 0,
    background: '#0d1117',
  },
  navGroup: {
    display: 'flex',
    gap: 4,
    marginRight: 10,
  },
  topbar: {
    position: 'absolute',
    top: 0, left: 0, right: 0,
    height: 44,
    background: 'rgba(0,0,0,0.65)',
    backdropFilter: 'blur(8px)',
    borderBottom: '1px solid rgba(255,255,255,0.07)',
    display: 'flex',
    alignItems: 'center',
    gap: 10,
    padding: '0 16px',
    zIndex: 10,
  },
  logo: {
    fontWeight: 800,
    fontSize: '1.05rem',
    letterSpacing: 2,
    background: 'linear-gradient(90deg, #00d2ff 0%, #7b2ff7 100%)',
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
  },
  sep: { width: 1, height: 20, background: 'rgba(255,255,255,0.12)' },
  sub: { color: 'rgba(255,255,255,0.35)', fontSize: 11, letterSpacing: 1 },
  dot: { width: 7, height: 7, borderRadius: '50%' },
  wsLabel: { fontSize: 11, color: 'rgba(255,255,255,0.45)', letterSpacing: 1 },
  navBtn: {
    background: 'transparent',
    border: '1px solid rgba(255,255,255,0.12)',
    borderRadius: 4,
    color: 'rgba(255,255,255,0.45)',
    fontSize: 11,
    letterSpacing: 1,
    padding: '3px 10px',
    cursor: 'pointer',
    transition: 'all 0.15s',
  },
  navBtnActive: {
    background: 'rgba(0,210,255,0.12)',
    borderColor: 'rgba(0,210,255,0.4)',
    color: '#00d2ff',
  },
  sidebar: {
    position: 'absolute',
    top: 52,
    right: 12,
    width: 300,
    display: 'flex',
    flexDirection: 'column',
    gap: 10,
    zIndex: 10,
    maxHeight: 'calc(100vh - 120px)',
    overflowY: 'auto',
  },
  panel: {
    background: 'rgba(0,0,0,0.72)',
    backdropFilter: 'blur(10px)',
    border: '1px solid rgba(255,255,255,0.08)',
    borderRadius: 8,
    padding: 14,
  },
  panelTitle: {
    fontSize: 10,
    fontWeight: 700,
    letterSpacing: 2,
    color: 'rgba(255,255,255,0.3)',
    textTransform: 'uppercase',
    marginBottom: 10,
  },
  feedBar: {
    position: 'absolute',
    bottom: 0, left: 0, right: 0,
    height: 140,
    background: 'rgba(0,0,0,0.72)',
    backdropFilter: 'blur(10px)',
    borderTop: '1px solid rgba(255,255,255,0.07)',
    padding: '8px 12px',
    zIndex: 10,
    overflow: 'hidden',
  },
}
