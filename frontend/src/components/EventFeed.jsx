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

export function EventFeed({ events, connected }) {
  const counts = {}
  for (const e of events) {
    counts[e.event_type] = (counts[e.event_type] || 0) + 1
  }

  return (
    <div style={styles.panel}>
      <div style={styles.header}>
        <span style={styles.title}>Live Events</span>
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
        {events.map(e => (
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
