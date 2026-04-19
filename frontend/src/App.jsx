import { useTelemetry } from './hooks/useTelemetry'
import { QueryPanel }     from './components/QueryPanel'
import { GraphPanel }     from './components/GraphPanel'
import { ChemicalPanel }  from './components/ChemicalPanel'
import { LoopPanel }      from './components/LoopPanel'
import { ProvenancePanel } from './components/ProvenancePanel'
import { EventFeed }      from './components/EventFeed'

export default function App() {
  const { events, connected } = useTelemetry()

  return (
    <div style={styles.root}>
      <header style={styles.topbar}>
        <span style={styles.logo}>CEREBRUM</span>
        <span style={styles.sep} />
        <span style={styles.sub}>Knowledge Graph Reasoning</span>
        <span style={{ flex: 1 }} />
        <span style={{ ...styles.dot, background: connected ? '#3fb950' : '#f85149' }} />
        <span style={styles.wsLabel}>{connected ? 'Live' : 'Offline'}</span>
        <a href="/v1/docs" style={styles.docsLink} target="_blank">API Docs</a>
      </header>

      <main style={styles.grid}>
        {/* Row 1 */}
        <Cell style={{ gridColumn: '1 / 3', gridRow: '1' }}>
          <QueryPanel />
        </Cell>
        <Cell style={{ gridColumn: '3 / 5', gridRow: '1' }}>
          <ChemicalPanel events={events} />
        </Cell>

        {/* Row 2 */}
        <Cell style={{ gridColumn: '1 / 3', gridRow: '2' }}>
          <GraphPanel events={events} />
        </Cell>
        <Cell style={{ gridColumn: '3', gridRow: '2' }}>
          <LoopPanel />
        </Cell>
        <Cell style={{ gridColumn: '4', gridRow: '2' }}>
          <ProvenancePanel />
        </Cell>

        {/* Row 3 — full width event feed */}
        <Cell style={{ gridColumn: '1 / 5', gridRow: '3', minHeight: 180 }}>
          <EventFeed events={events} connected={connected} />
        </Cell>
      </main>
    </div>
  )
}

function Cell({ children, style }) {
  return <div style={{ ...styles.cell, ...style }}>{children}</div>
}

const styles = {
  root: {
    minHeight: '100vh',
    background: '#0d1117',
    color: '#c9d1d9',
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif',
    fontSize: 14,
    display: 'flex',
    flexDirection: 'column',
  },
  topbar: {
    height: 48,
    background: '#161b22',
    borderBottom: '1px solid #30363d',
    display: 'flex',
    alignItems: 'center',
    gap: 12,
    padding: '0 16px',
    position: 'sticky',
    top: 0,
    zIndex: 100,
  },
  logo: {
    fontWeight: 800,
    fontSize: '1.1rem',
    background: 'linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%)',
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
  },
  sep: { width: 1, height: 24, background: '#30363d' },
  sub: { color: '#8b949e', fontSize: 12 },
  dot: { width: 8, height: 8, borderRadius: '50%' },
  wsLabel: { fontSize: 12, color: '#8b949e' },
  docsLink: { fontSize: 12, color: '#58a6ff', textDecoration: 'none', padding: '3px 8px', border: '1px solid #30363d', borderRadius: 4 },
  grid: {
    flex: 1,
    display: 'grid',
    gridTemplateColumns: 'repeat(4, 1fr)',
    gridTemplateRows: '220px 320px auto',
    gap: 8,
    padding: 8,
  },
  cell: {
    background: '#161b22',
    border: '1px solid #21262d',
    borderRadius: 8,
    padding: 12,
    overflow: 'hidden',
  },
}
