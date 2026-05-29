import { useState } from 'react'
import { api } from '../api'

export function QueryPanel({ onAnswers }) {
  const [query, setQuery] = useState('')
  const [answers, setAnswers] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const submit = async () => {
    if (!query.trim()) return
    setLoading(true)
    setError(null)
    try {
      const res = await api.query(query)
      const paths = res.data.paths || []
      setAnswers(paths)
      if (onAnswers) onAnswers(paths)
    } catch (e) {
      setError(e.response?.data?.detail || e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={styles.panel}>
      <div style={styles.title}>Query</div>
      <div style={styles.inputRow}>
        <input
          style={styles.input}
          value={query}
          onChange={e => setQuery(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && submit()}
          placeholder="Ask the brain…"
        />
        <button style={styles.btn} onClick={submit} disabled={loading}>
          {loading ? '…' : 'Run'}
        </button>
      </div>
      {error && <div style={styles.error}>{error}</div>}
      <div style={styles.results}>
        {answers.map((p, i) => (
          <div key={i} style={styles.answer}>
            <span style={styles.rank}>#{p.rank ?? i + 1}</span>
            <span style={styles.answerText}>{p.answer_entity}</span>
            <span style={styles.score}>{(p.score ?? 0).toFixed(3)}</span>
          </div>
        ))}
        {!loading && answers.length === 0 && (
          <div style={styles.empty}>No results yet</div>
        )}
      </div>
    </div>
  )
}

const styles = {
  panel: { height: '100%', display: 'flex', flexDirection: 'column', gap: 8 },
  title: { fontWeight: 700, fontSize: 12, color: '#8b949e', textTransform: 'uppercase', letterSpacing: 1 },
  inputRow: { display: 'flex', gap: 6 },
  input: {
    flex: 1, padding: '6px 10px', background: '#0d1117', border: '1px solid #30363d',
    borderRadius: 6, color: '#c9d1d9', fontSize: 13, outline: 'none',
  },
  btn: {
    padding: '6px 14px', background: '#238636', border: 'none', borderRadius: 6,
    color: '#fff', cursor: 'pointer', fontSize: 13, fontWeight: 600,
  },
  error: { color: '#f85149', fontSize: 12 },
  results: { flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 4 },
  answer: {
    display: 'flex', alignItems: 'center', gap: 8, padding: '5px 8px',
    background: '#161b22', borderRadius: 5, border: '1px solid #21262d',
  },
  rank: { color: '#58a6ff', fontSize: 11, minWidth: 24 },
  answerText: { color: '#c9d1d9', fontSize: 13, flex: 1 },
  score: { color: '#8b949e', fontSize: 10 },
  empty: { color: '#8b949e', fontSize: 12, textAlign: 'center', marginTop: 16 },
}
