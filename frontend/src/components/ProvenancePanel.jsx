import { useState, useEffect } from 'react'
import { api } from '../api'

export function ProvenancePanel() {
  const [stats, setStats]   = useState(null)
  const [batches, setBatches] = useState([])
  const [rolling, setRolling] = useState(null)

  const refresh = () => {
    api.provenanceStats().then(r => setStats(r.data)).catch(() => {})
    api.provenanceBatches().then(r => setBatches(r.data?.batches || [])).catch(() => {})
  }

  useEffect(() => {
    refresh()
    const t = setInterval(refresh, 8000)
    return () => clearInterval(t)
  }, [])

  const rollback = async (batchId) => {
    if (!window.confirm(`Rollback batch ${batchId}?`)) return
    setRolling(batchId)
    try {
      await api.rollbackBatch(batchId)
      setTimeout(refresh, 400)
    } catch (e) {
      alert(e.response?.data?.detail || e.message)
    } finally {
      setRolling(null)
    }
  }

  return (
    <div style={styles.panel}>
      <div style={styles.title}>Provenance</div>

      {stats && (
        <div style={styles.statsRow}>
          <StatPill label="Batches"  value={stats.total_batches ?? 0} />
          <StatPill label="Edges"    value={stats.total_edges_recorded ?? 0} />
          <StatPill label="Rollbacks" value={stats.total_rollbacks ?? 0} color="#f85149" />
        </div>
      )}

      <div style={styles.table}>
        <div style={styles.thead}>
          <span style={{ flex: 2 }}>Batch ID</span>
          <span style={{ flex: 1, textAlign: 'center' }}>Edges</span>
          <span style={{ flex: 1, textAlign: 'center' }}>Status</span>
          <span style={{ flex: 1 }} />
        </div>
        {batches.slice(0, 12).map(b => (
          <div key={b.batch_id} style={styles.row}>
            <span style={{ flex: 2, color: '#58a6ff', fontSize: 11, overflow: 'hidden', textOverflow: 'ellipsis' }}>
              {b.batch_id?.slice(0, 12)}…
            </span>
            <span style={{ flex: 1, textAlign: 'center', color: '#c9d1d9' }}>{b.edge_count ?? 0}</span>
            <span style={{ flex: 1, textAlign: 'center', color: b.rolled_back ? '#f85149' : '#3fb950', fontSize: 11 }}>
              {b.rolled_back ? 'rolled back' : 'active'}
            </span>
            <span style={{ flex: 1, textAlign: 'right' }}>
              {!b.rolled_back && (
                <button
                  style={styles.rollBtn}
                  onClick={() => rollback(b.batch_id)}
                  disabled={rolling === b.batch_id}
                >
                  {rolling === b.batch_id ? '…' : 'Rollback'}
                </button>
              )}
            </span>
          </div>
        ))}
        {batches.length === 0 && (
          <div style={styles.empty}>No batches recorded</div>
        )}
      </div>
    </div>
  )
}

function StatPill({ label, value, color }) {
  return (
    <div style={styles.pill}>
      <span style={styles.pillLabel}>{label}</span>
      <span style={{ ...styles.pillValue, color: color || '#c9d1d9' }}>{value}</span>
    </div>
  )
}

const styles = {
  panel: { height: '100%', display: 'flex', flexDirection: 'column', gap: 8 },
  title: { fontWeight: 700, fontSize: 12, color: '#8b949e', textTransform: 'uppercase', letterSpacing: 1 },
  statsRow: { display: 'flex', gap: 6 },
  pill: { flex: 1, background: '#161b22', borderRadius: 5, padding: '5px 8px', display: 'flex', flexDirection: 'column' },
  pillLabel: { fontSize: 10, color: '#8b949e' },
  pillValue: { fontSize: 16, fontWeight: 700 },
  table: { flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 2 },
  thead: { display: 'flex', padding: '2px 6px', fontSize: 10, color: '#8b949e', textTransform: 'uppercase' },
  row: {
    display: 'flex', alignItems: 'center', padding: '4px 6px',
    background: '#0d1117', borderRadius: 4, fontSize: 12,
  },
  rollBtn: {
    padding: '2px 8px', background: 'transparent', border: '1px solid #da3633',
    borderRadius: 4, color: '#f85149', cursor: 'pointer', fontSize: 11,
  },
  empty: { color: '#8b949e', fontSize: 12, textAlign: 'center', marginTop: 12 },
}
