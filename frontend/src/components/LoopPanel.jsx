import { useState, useEffect } from 'react'
import { api } from '../api'

export function LoopPanel() {
  const [status, setStatus] = useState(null)
  const [loading, setLoading] = useState(false)

  const refresh = () => {
    api.loopStatus().then(r => setStatus(r.data)).catch(() => {})
  }

  useEffect(() => {
    refresh()
    const t = setInterval(refresh, 5000)
    return () => clearInterval(t)
  }, [])

  const toggle = async () => {
    if (!status) return
    setLoading(true)
    try {
      if (status.running) await api.loopStop()
      else await api.loopStart()
      setTimeout(refresh, 500)
    } finally {
      setLoading(false)
    }
  }

  const toggleFeature = async (feature, current) => {
    setLoading(true)
    try {
      const config = {}
      if (feature === 'active_inference') config.active_inference = !current
      if (feature === 'gui_adaptation') config.gui_adaptation = !current
      if (feature === 'autonomous_research') config.autonomous_research = !current
      if (feature === 'recursive_synthesis') config.recursive_synthesis = !current
      if (feature === 'metaplasticity') config.metaplasticity = !current
      
      await api.loopConfigure(config)
      refresh()
    } finally {
      setLoading(false)
    }
  }

  const running = status?.running ?? false
  const rate = status?.current_approval_rate != null ? (status.current_approval_rate * 100).toFixed(1) : '–'
  const tripped = status?.circuit_breaker_tripped ?? false
  const cycles = status?.total_cycles ?? 0

  return (
    <div style={styles.panel}>
      <div style={styles.header}>
        <span style={styles.title}>Discovery Loop</span>
        <button
          style={{ ...styles.btn, background: running ? '#da3633' : '#238636' }}
          onClick={toggle}
          disabled={loading}
        >
          {loading ? '…' : running ? 'Stop' : 'Start'}
        </button>
      </div>

      <div style={styles.stats}>
        <Stat label="Status"      value={running ? '● Running' : '○ Stopped'} color={running ? '#3fb950' : '#8b949e'} />
        <Stat label="Cycles"      value={cycles} />
        <Stat label="Approval"    value={`${rate}%`} />
        <Stat label="Breaker"     value={tripped ? '⚡ Tripped' : 'OK'} color={tripped ? '#f85149' : '#3fb950'} />
      </div>

      <div style={styles.featureList}>
        <FeatureToggle 
          label="Active Inference" 
          active={status?.active_inference_enabled} 
          onToggle={() => toggleFeature('active_inference', status.active_inference_enabled)} 
        />
        <FeatureToggle 
          label="GUI Adaptation" 
          active={status?.gui_adaptation_enabled} 
          onToggle={() => toggleFeature('gui_adaptation', status.gui_adaptation_enabled)} 
        />
        <FeatureToggle 
          label="Autonomous Research" 
          active={status?.autonomous_research_enabled} 
          onToggle={() => toggleFeature('autonomous_research', status.autonomous_research_enabled)} 
        />
        <FeatureToggle 
          label="Recursive Synthesis" 
          active={status?.recursive_synthesis_enabled} 
          onToggle={() => toggleFeature('recursive_synthesis', status.recursive_synthesis_enabled)} 
        />
        <FeatureToggle 
          label="Metaplasticity" 
          active={status?.metaplasticity_enabled} 
          onToggle={() => toggleFeature('metaplasticity', status.metaplasticity_enabled)} 
        />
      </div>

      {status?.recent_cycles?.length > 0 && (
        <div style={styles.cycleList}>
          {status.recent_cycles.slice(-5).reverse().map((c, i) => (
            <div key={i} style={styles.cycle}>
              <span style={styles.cycleNum}>#{c.cycle_number ?? i + 1}</span>
              <span style={styles.cycleInfo}>
                {c.candidates_found ?? 0} found · {c.materialized ?? 0} materialized
              </span>
              {c.circuit_breaker_tripped && <span style={{ color: '#f85149' }}>⚡</span>}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function Stat({ label, value, color }) {
  return (
    <div style={styles.stat}>
      <span style={styles.statLabel}>{label}</span>
      <span style={{ ...styles.statValue, color: color || '#c9d1d9' }}>{value}</span>
    </div>
  )
}

function FeatureToggle({ label, active, onToggle }) {
  return (
    <div style={styles.feature} onClick={onToggle}>
      <div style={{ ...styles.toggle, background: active ? '#238636' : '#30363d' }}>
        <div style={{ ...styles.knob, transform: active ? 'translateX(12px)' : 'translateX(0)' }} />
      </div>
      <span style={styles.featureLabel}>{label}</span>
    </div>
  )
}

const styles = {
  panel: { height: '100%', display: 'flex', flexDirection: 'column', gap: 8 },
  header: { display: 'flex', justifyContent: 'space-between', alignItems: 'center' },
  title: { fontWeight: 700, fontSize: 12, color: '#8b949e', textTransform: 'uppercase', letterSpacing: 1 },
  btn: { padding: '4px 12px', border: 'none', borderRadius: 5, color: '#fff', cursor: 'pointer', fontSize: 12, fontWeight: 600 },
  stats: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 },
  stat: { background: '#161b22', borderRadius: 5, padding: '6px 8px', display: 'flex', flexDirection: 'column', gap: 2 },
  statLabel: { fontSize: 10, color: '#8b949e', textTransform: 'uppercase' },
  statValue: { fontSize: 14, fontWeight: 700 },
  featureList: { display: 'flex', flexDirection: 'column', gap: 4, margin: '4px 0' },
  feature: { display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer', padding: '2px 0' },
  featureLabel: { fontSize: 11, color: '#c9d1d9' },
  toggle: { width: 28, height: 16, borderRadius: 14, padding: 2, transition: 'background 0.2s' },
  knob: { width: 12, height: 12, borderRadius: '50%', background: '#fff', transition: 'transform 0.2s' },
  cycleList: { flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 3 },
  cycle: { display: 'flex', alignItems: 'center', gap: 6, padding: '3px 6px', background: '#0d1117', borderRadius: 4, fontSize: 11 },
  cycleNum: { color: '#58a6ff', minWidth: 28 },
  cycleInfo: { flex: 1, color: '#8b949e' },
}
