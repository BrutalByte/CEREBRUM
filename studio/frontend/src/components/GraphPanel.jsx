import { useEffect, useRef, useState } from 'react'
import cytoscape from 'cytoscape'
import { api } from '../api'

// Community → color (matches Python golden-ratio HSV palette)
const communityColor = (cid) => {
  const hue = ((cid * 137.508) % 360)
  return `hsl(${hue}, 70%, 55%)`
}

export function GraphPanel({ events }) {
  const containerRef = useRef(null)
  const cyRef = useRef(null)
  const [nodeCount, setNodeCount] = useState(0)
  const [edgeCount, setEdgeCount] = useState(0)

  // Build graph on mount
  useEffect(() => {
    if (!containerRef.current) return

    let cy = cytoscape({
      container: containerRef.current,
      style: [
        {
          selector: 'node',
          style: {
            'background-color': (ele) => communityColor(ele.data('community') || 0),
            'label': 'data(label)',
            'color': '#c9d1d9',
            'font-size': 9,
            'text-valign': 'bottom',
            'text-margin-y': 4,
            width: 18, height: 18,
          },
        },
        {
          selector: 'edge',
          style: {
            'line-color': '#30363d',
            'target-arrow-color': '#30363d',
            'target-arrow-shape': 'triangle',
            'curve-style': 'bezier',
            'width': (ele) => 1 + (ele.data('weight') || 0.5) * 2,
            'opacity': 0.6,
          },
        },
        {
          selector: '.pulsed',
          style: { 'line-color': '#58a6ff', 'target-arrow-color': '#58a6ff', opacity: 1 },
        },
        {
          selector: '.active-node',
          style: { 'background-color': '#58a6ff', 'border-width': 2, 'border-color': '#fff' },
        },
      ],
      layout: { name: 'cose', animate: false },
      userZoomingEnabled: true,
      userPanningEnabled: true,
    })
    cyRef.current = cy

    // Load graph data
    Promise.all([api.edges(1000), api.communities()]).then(([edgesRes, commRes]) => {
      const communityMap = commRes.data?.communities || {}
      const elements = []
      const nodesSeen = new Set()

      for (const edge of (edgesRes.data?.edges || [])) {
        const src = edge.source_id || edge.source
        const tgt = edge.target_id || edge.target
        if (!src || !tgt) continue
        if (!nodesSeen.has(src)) {
          nodesSeen.add(src)
          elements.push({ data: { id: src, label: src.replace(/_/g, ' '), community: communityMap[src] ?? 0 } })
        }
        if (!nodesSeen.has(tgt)) {
          nodesSeen.add(tgt)
          elements.push({ data: { id: tgt, label: tgt.replace(/_/g, ' '), community: communityMap[tgt] ?? 0 } })
        }
        elements.push({
          data: {
            id: `${src}::${edge.relation_type}::${tgt}`,
            source: src, target: tgt,
            weight: edge.weight ?? 0.5,
          },
        })
      }

      cy.add(elements)
      cy.layout({ name: 'cose', animate: false, randomize: false }).run()
      setNodeCount(nodesSeen.size)
      setEdgeCount(edgesRes.data?.edges?.length || 0)
    }).catch(() => {})

    return () => { cy.destroy(); cyRef.current = null }
  }, [])

  // Flash edges on SYNAPTIC_PULSE events
  useEffect(() => {
    const pulse = events.find(e => e.event_type === 'SYNAPTIC_PULSE')
    if (!pulse || !cyRef.current) return
    const { source_node, target_node } = pulse.payload || {}
    if (!source_node || !target_node) return
    const cy = cyRef.current
    const edges = cy.edges().filter(e =>
      e.data('source') === source_node && e.data('target') === target_node
    )
    edges.addClass('pulsed')
    cy.$id(source_node).addClass('active-node')
    cy.$id(target_node).addClass('active-node')
    setTimeout(() => {
      edges.removeClass('pulsed')
      cy.$id(source_node).removeClass('active-node')
      cy.$id(target_node).removeClass('active-node')
    }, 1800)
  }, [events])

  return (
    <div style={styles.panel}>
      <div style={styles.header}>
        <span style={styles.title}>Graph</span>
        <span style={styles.badge}>{nodeCount} nodes · {edgeCount} edges</span>
      </div>
      <div ref={containerRef} style={styles.canvas} />
    </div>
  )
}

const styles = {
  panel: { height: '100%', display: 'flex', flexDirection: 'column' },
  header: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 },
  title: { fontWeight: 700, fontSize: 12, color: '#8b949e', textTransform: 'uppercase', letterSpacing: 1 },
  badge: { fontSize: 11, color: '#8b949e' },
  canvas: { flex: 1, background: '#0d1117', borderRadius: 6, border: '1px solid #21262d' },
}
