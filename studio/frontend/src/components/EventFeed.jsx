const TYPE_COLORS = {
  SYNAPTIC_PULSE:      '#58a6ff',
  NEUROGENESIS:        '#3fb950',
  SYNAPTOGENESIS:      '#3fb950',
  SYNAPTIC_PRUNE:      '#f85149',
  DISSONANCE:          '#d29922',
  METABOLIC_FLUX:      '#bc8cff',
  DEFAULT_MODE_PULSE:  '#79c0ff',
  VALENCE_UPDATE:      '#ffa657',
  SYNAPTIC_DECAY:      '#8b949e',
  CONSOLIDATION_PULSE: '#56d364',
  CORTICAL_GLOW:       '#e3b341',
  GOAL_UPDATE:         '#f0883e',
}

const TYPE_SHORT = {
  SYNAPTIC_PULSE:      'PULSE',
  NEUROGENESIS:        'GENESIS',
  SYNAPTOGENESIS:      'SYNAP',
  SYNAPTIC_PRUNE:      'PRUNE',
  DISSONANCE:          'DISS',
  METABOLIC_FLUX:      'FLUX',
  DEFAULT_MODE_PULSE:  'DMN',
  VALENCE_UPDATE:      'VALENCE',
  SYNAPTIC_DECAY:      'DECAY',
  CONSOLIDATION_PULSE: 'CONSOL',
  CORTICAL_GLOW:       'GLOW',
  GOAL_UPDATE:         'GOAL',
}

import { useState } from 'react'

const RESEARCH_TYPES = [
  'NEUROGENESIS', 'SYNAPTOGENESIS', 'CONSOLIDATION_PULSE', 
  'HYPOTHESIS_GENERATED', 'PROPOSAL_CREATED', 'REM_CYCLE_START', 'REM_CYCLE_END'
]

export function EventFeed({ events, connected }) {
  const [filter, setFilter] = useState('all') // 'all' | 'research'

  const filtered = filter === 'all' 
    ? events 
    : events.filter(e => RESEARCH_TYPES.includes(e.event_type) || e.event_type.includes('PROPOSAL') || e.event_type.includes('HYPOTHESIS'))

  const counts = {}
  for (const e of events) {
    counts[e.event_type] = (counts[e.event_type] || 0) + 1
  }

  return (
    <div style={styles.panel}>
      <div style={styles.header}>
        <span style={styles.title}>Live Events</span>
        
        <div style={styles.filterGroup}>
          <button 
            style={{ ...styles.filterBtn, ...(filter === 'all' ? styles.filterBtnActive : {}) }}
            onClick={() => setFilter('all')}
          >
            All
          </button>
          <button 
            style={{ ...styles.filterBtn, ...(filter === 'research' ? styles.filterBtnActive : {}) }}
            onClick={() => setFilter('research')}
          >
            Research
          </button>
        </div>

        <span style={{ ...styles.dot, background: connected ? '#3fb950' : '#f85149' }} />
        <span style={styles.status}>{connected ? 'Connected' : 'Disconnected'}</span>
      </div>

      <div style={styles.counters}>
        {Object.entries(counts).map(([type, n]) => (
          <span key={type} style={{ ...styles.badge, borderColor: TYPE_COLORS[type] || '#30363d' }}>
            <span style={{ color: TYPE_COLORS[type] || '#8b949e' }}>{TYPE_SHORT[type] || type}</span>
            <span style={styles.badgeCount}>{n}</span>
          </span>
        ))}
      </div>

      <div style={styles.feed}>
        {filtered.map(e => (
          <div key={e._id} style={styles.row}>
            <span style={{ ...styles.tag, color: TYPE_COLORS[e.event_type] || '#8b949e' }}>
              {TYPE_SHORT[e.event_type] || e.event_type}
            </span>
            <span style={styles.detail}>
              {e.payload?.source_node && `${e.payload.source_node} → ${e.payload.target_node}`}
              {e.payload?.node_id && e.payload.node_id}
              {e.payload?.insight_count != null && `${e.payload.insight_count} insights`}
              {e.payload?.edges_decayed != null && `${e.payload.edges_decayed} decayed`}
            </span>
            <span style={styles.ts}>{new Date(e.header?.timestamp * 1000).toLocaleTimeString()}</span>
          </div>
        ))}
        {filtered.length === 0 && (
          <div style={{ color: '#484f58', fontSize: 11, textAlign: 'center', marginTop: 10 }}>
            No {filter} events in buffer
          </div>
        )}
      </div>
    </div>
  )
}

const styles = {
  panel: { height: '100%', display: 'flex', flexDirection: 'column', gap: 6 },
  header: { display: 'flex', alignItems: 'center', gap: 6 },
  title: { fontWeight: 700, fontSize: 12, color: '#8b949e', textTransform: 'uppercase', letterSpacing: 1, flex: 1 },
  dot: { width: 8, height: 8, borderRadius: '50%' },
  status: { fontSize: 11, color: '#8b949e' },
  filterGroup: { display: 'flex', gap: 2, background: 'rgba(255,255,255,0.05)', padding: 2, borderRadius: 4, marginRight: 8 },
  filterBtn: { 
    background: 'transparent', border: 'none', borderRadius: 3, 
    color: '#8b949e', fontSize: 10, padding: '2px 6px', cursor: 'pointer',
  },
  filterBtnActive: { background: 'rgba(255,255,255,0.1)', color: '#e6edf3' },
  counters: { display: 'flex', flexWrap: 'wrap', gap: 4 },
  badge: {
    display: 'inline-flex', alignItems: 'center', gap: 4, padding: '2px 6px',
    border: '1px solid', borderRadius: 4, fontSize: 10,
  },
  badgeCount: { color: '#c9d1d9', fontWeight: 600 },
  feed: { flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 2 },
  row: {
    display: 'flex', alignItems: 'center', gap: 6, padding: '3px 6px',
    borderRadius: 4, background: '#0d1117', fontSize: 11,
  },
  tag: { minWidth: 52, fontWeight: 600 },
  detail: { flex: 1, color: '#8b949e', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' },
  ts: { color: '#484f58', minWidth: 64, textAlign: 'right' },
}
