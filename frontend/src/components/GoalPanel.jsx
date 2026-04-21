import { useState, useEffect } from 'react'
import { api } from '../api'

export function GoalPanel() {
  const [goals, setGoals] = useState([])
  const [newGoal, setNewGoal] = useState('')
  const [loading, setLoading] = useState(false)

  const refresh = () => {
    api.goals().then(r => setGoals(r.data?.goals || [])).catch(() => {})
  }

  useEffect(() => {
    refresh()
    const t = setInterval(refresh, 5000)
    return () => clearInterval(t)
  }, [])

  const submit = async (e) => {
    e.preventDefault()
    if (!newGoal.trim()) return
    setLoading(true)
    try {
      await api.addGoal(newGoal)
      setNewGoal('')
      refresh()
    } catch (e) {
      alert(e.response?.data?.detail || e.message)
    } finally {
      setLoading(false)
    }
  }

  const remove = async (id) => {
    try {
      await api.deleteGoal(id)
      refresh()
    } catch (e) {
      alert(e.response?.data?.detail || e.message)
    }
  }

  return (
    <div style={styles.panel}>
      <div style={styles.title}>Goal System</div>
      
      <form onSubmit={submit} style={styles.form}>
        <input
          style={styles.input}
          placeholder="New goal..."
          value={newGoal}
          onChange={e => setNewGoal(e.target.value)}
          disabled={loading}
        />
        <button type="submit" style={styles.addBtn} disabled={loading || !newGoal.trim()}>
          +
        </button>
      </form>

      <div style={styles.list}>
        {goals.map(g => (
          <div key={g.id} style={styles.goal}>
            <div style={styles.goalMain}>
              <span style={styles.goalText}>{g.text}</span>
              <span style={styles.goalMeta}>
                P:{g.priority} · {g.status}
              </span>
            </div>
            <button style={styles.delBtn} onClick={() => remove(g.id)}>✕</button>
          </div>
        ))}
        {goals.length === 0 && (
          <div style={styles.empty}>No active goals</div>
        )}
      </div>
    </div>
  )
}

const styles = {
  panel: { height: '100%', display: 'flex', flexDirection: 'column', gap: 10 },
  title: { fontWeight: 700, fontSize: 12, color: '#8b949e', textTransform: 'uppercase', letterSpacing: 1 },
  form: { display: 'flex', gap: 6 },
  input: {
    flex: 1, background: '#0d1117', border: '1px solid #30363d', borderRadius: 4,
    color: '#c9d1d9', padding: '4px 8px', fontSize: 12, outline: 'none',
  },
  addBtn: {
    padding: '0 10px', background: '#238636', border: 'none', borderRadius: 4,
    color: '#fff', cursor: 'pointer', fontSize: 16, fontWeight: 700,
  },
  list: { flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 4 },
  goal: {
    background: '#161b22', borderRadius: 6, padding: '6px 10px',
    display: 'flex', alignItems: 'center', gap: 8, border: '1px solid transparent',
  },
  goalMain: { flex: 1, display: 'flex', flexDirection: 'column' },
  goalText: { fontSize: 12, color: '#e6edf3' },
  goalMeta: { fontSize: 9, color: '#8b949e', textTransform: 'uppercase' },
  delBtn: { background: 'none', border: 'none', color: '#8b949e', cursor: 'pointer', fontSize: 12, padding: 4 },
  empty: { color: '#8b949e', fontSize: 12, textAlign: 'center', marginTop: 10 },
}
