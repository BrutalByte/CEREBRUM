import { useState, useEffect } from 'react'
import { RadarChart, PolarGrid, PolarAngleAxis, Radar, ResponsiveContainer, Tooltip } from 'recharts'
import { api } from '../api'

const SCALARS = ['Reinforcement', 'Arousal', 'Novelty', 'Cohesion', 'Persistence']
const COLORS   = ['#3fb950', '#d29922', '#58a6ff', '#bc8cff', '#f85149']

const empty = () => SCALARS.map(name => ({ name, value: 0 }))

export function ChemicalPanel({ events }) {
  const [data, setData] = useState(empty())

  // Update on METABOLIC_FLUX websocket events
  useEffect(() => {
    const flux = events.find(e => e.event_type === 'METABOLIC_FLUX')
    if (!flux) return
    const s = flux.payload?.state || {}
    setData([
      { name: 'Reinforcement', value: +(s.reinforcement ?? s.Reinforcement ?? 0).toFixed(3) },
      { name: 'Arousal',       value: +(s.arousal       ?? s.Arousal       ?? 0).toFixed(3) },
      { name: 'Novelty',       value: +(s.novelty       ?? s.Novelty       ?? 0).toFixed(3) },
      { name: 'Cohesion',      value: +(s.cohesion      ?? s.Cohesion      ?? 0).toFixed(3) },
      { name: 'Persistence',   value: +(s.persistence   ?? s.Persistence   ?? 0).toFixed(3) },
    ])
  }, [events])

  // Also poll REST /chemical on mount
  useEffect(() => {
    api.chemical().then(r => {
      const s = r.data
      setData([
        { name: 'Reinforcement', value: +(s.reinforcement ?? 0).toFixed(3) },
        { name: 'Arousal',       value: +(s.arousal       ?? 0).toFixed(3) },
        { name: 'Novelty',       value: +(s.novelty       ?? 0).toFixed(3) },
        { name: 'Cohesion',      value: +(s.cohesion      ?? 0).toFixed(3) },
        { name: 'Persistence',   value: +(s.persistence   ?? 0).toFixed(3) },
      ])
    }).catch(() => {})
  }, [])

  return (
    <div style={styles.panel}>
      <div style={styles.title}>Chemical Modulator</div>
      <ResponsiveContainer width="100%" height={180}>
        <RadarChart data={data} margin={{ top: 5, right: 20, bottom: 5, left: 20 }}>
          <PolarGrid stroke="#30363d" />
          <PolarAngleAxis dataKey="name" tick={{ fill: '#8b949e', fontSize: 11 }} />
          <Radar dataKey="value" stroke="#58a6ff" fill="#58a6ff" fillOpacity={0.25} />
          <Tooltip
            contentStyle={{ background: '#161b22', border: '1px solid #30363d', borderRadius: 6 }}
            labelStyle={{ color: '#c9d1d9' }}
            itemStyle={{ color: '#58a6ff' }}
          />
        </RadarChart>
      </ResponsiveContainer>
      <div style={styles.bars}>
        {data.map((d, i) => (
          <div key={d.name} style={styles.barRow}>
            <span style={{ ...styles.label, color: COLORS[i] }}>{d.name.slice(0, 4)}</span>
            <div style={styles.track}>
              <div style={{ ...styles.fill, width: `${d.value * 100}%`, background: COLORS[i] }} />
            </div>
            <span style={styles.val}>{d.value.toFixed(2)}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

const styles = {
  panel: { height: '100%', display: 'flex', flexDirection: 'column', gap: 6 },
  title: { fontWeight: 700, fontSize: 12, color: '#8b949e', textTransform: 'uppercase', letterSpacing: 1 },
  bars: { display: 'flex', flexDirection: 'column', gap: 4 },
  barRow: { display: 'flex', alignItems: 'center', gap: 6 },
  label: { fontSize: 10, minWidth: 28, fontWeight: 600 },
  track: { flex: 1, height: 6, background: '#21262d', borderRadius: 3, overflow: 'hidden' },
  fill: { height: '100%', borderRadius: 3, transition: 'width 0.4s ease' },
  val: { fontSize: 10, color: '#8b949e', minWidth: 28, textAlign: 'right' },
}
