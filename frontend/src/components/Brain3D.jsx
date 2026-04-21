import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls }   from 'three/addons/controls/OrbitControls.js'
import { EffectComposer }  from 'three/addons/postprocessing/EffectComposer.js'
import { RenderPass }      from 'three/addons/postprocessing/RenderPass.js'
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js'
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js'
import { api } from '../api'

// ── Constants ────────────────────────────────────────────────────────────────
const COLOR_QUERY_NODE = new THREE.Color(1.0,  0.85, 0.1)
const COLOR_QUERY_EDGE = new THREE.Color(0.0,  0.9,  1.0)
const COLOR_NEW_EDGE   = new THREE.Color(0.1,  1.0,  0.4)
const COLOR_WORMHOLE   = new THREE.Color(0.7,  0.1,  1.0)
const COLOR_DIM        = new THREE.Color(0.03, 0.03, 0.03)
const EDGE_LIMIT       = 10000
const MAX_NODES        = 10000
const MAX_EDGES        = 20000

const COLOR_SCHEMES = {
  classic:    (cid) => new THREE.Color().setHSL(((cid * 137.5) % 360) / 360, 0.7, 0.5),
  neon:       (cid) => new THREE.Color().setHSL(((cid * 60) % 360) / 360, 1.0, 0.6),
  monochrome: (cid) => new THREE.Color().setHSL(0, 0, 0.2 + (cid % 10) * 0.07),
  cyber: (cid) => {
    const palette = ['#00f2ff', '#00ff9f', '#ff0055', '#7b2ff7', '#fdfd96', '#ffb86c']
    return new THREE.Color(palette[cid % palette.length])
  }
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function getCommunityColor(cid, scheme = 'classic') {
  const fn = COLOR_SCHEMES[scheme] || COLOR_SCHEMES.classic
  return fn(cid)
}

function fibonacciSphere(n, radius) {
  const phi = Math.PI * (3 - Math.sqrt(5))
  return Array.from({ length: n }, (_, i) => {
    const y = 1 - (i / Math.max(n - 1, 1)) * 2
    const r = Math.sqrt(Math.max(0, 1 - y * y))
    const theta = phi * i
    return new THREE.Vector3(r * Math.cos(theta) * radius, y * radius, r * Math.sin(theta) * radius)
  })
}

// Community-clustered: place community centroids on outer sphere,
// members in a local sub-sphere around each centroid.
function communityClusteredLayout(nodeIds, communityMap) {
  const groups = {}
  for (const id of nodeIds) {
    const cid = communityMap[id] ?? 0
    if (!groups[cid]) groups[cid] = []
    groups[cid].push(id)
  }
  const cidList = Object.keys(groups)
  const maxSize = Math.max(...cidList.map(c => groups[c].length))
  const centroids = fibonacciSphere(cidList.length, 400)
  const posMap = {}
  cidList.forEach((cid, ci) => {
    const center  = centroids[ci]
    const members = groups[cid]
    const innerR  = 10 + Math.sqrt(members.length / maxSize) * 70
    if (members.length === 1) {
      posMap[members[0]] = center.clone()
    } else {
      const local = fibonacciSphere(members.length, innerR)
      members.forEach((id, mi) => { posMap[id] = center.clone().add(local[mi]) })
    }
  })
  return posMap
}

function neighborhoodLayout(nodeIds, nodeEdges, focalNodeId) {
  const posMap = {}
  if (!focalNodeId || !nodeEdges[focalNodeId]) {
    const pts = fibonacciSphere(nodeIds.length, 450)
    nodeIds.forEach((id, i) => posMap[id] = pts[i])
    return posMap
  }

  const visited = new Set([focalNodeId])
  posMap[focalNodeId] = new THREE.Vector3(0, 0, 0)

  // Layer 1: direct neighbors -> Sphere Shell 1
  const l1 = (nodeEdges[focalNodeId] || []).map(e => e.node).filter(id => !visited.has(id))
  l1.forEach(id => visited.add(id))
  const l1Pts = fibonacciSphere(l1.length, 180)
  l1.forEach((id, i) => posMap[id] = l1Pts[i])

  // Layer 2: neighbors of neighbors -> Sphere Shell 2
  const l2 = []
  l1.forEach(u => {
    (nodeEdges[u] || []).forEach(e => {
      if (!visited.has(e.node)) {
        visited.add(e.node)
        l2.push(e.node)
      }
    })
  })
  const l2Pts = fibonacciSphere(l2.length, 380)
  l2.forEach((id, i) => posMap[id] = l2Pts[i])

  // Remaining nodes: outer boundary shell
  const remaining = nodeIds.filter(id => !visited.has(id))
  const outerPts = fibonacciSphere(remaining.length, 700)
  remaining.forEach((id, i) => posMap[id] = outerPts[i])

  return posMap
}

function corticalColumnLayout(nodeIds, communityMap) {
  const groups = {}
  for (const id of nodeIds) {
    const cid = communityMap[id] ?? 0
    if (!groups[cid]) groups[cid] = []
    groups[cid].push(id)
  }
  
  const cidList = Object.keys(groups)
  const posMap = {}
  const ringRadius = 500
  
  cidList.forEach((cid, ci) => {
    const angle = (ci / cidList.length) * Math.PI * 2
    const centerX = Math.cos(angle) * ringRadius
    const centerZ = Math.sin(angle) * ringRadius
    
    const members = groups[cid]
    members.forEach((id, mi) => {
      // Stack nodes vertically in a column (cylinder)
      const height = (mi - (members.length / 2)) * 25
      const radialOffset = 40 + Math.random() * 20
      const innerAngle = Math.random() * Math.PI * 2
      
      posMap[id] = new THREE.Vector3(
        centerX + Math.cos(innerAngle) * radialOffset,
        height,
        centerZ + Math.sin(innerAngle) * radialOffset
      )
    })
  })
  return posMap
}

function hierarchicalLayout(nodeIds, communityMap, commCentroids) {
  const posMap = {}
  // Level 1: Community centroids on a large sphere
  // Level 2: Nodes distributed between centroid and origin based on importance (degree)
  // For now, simple implementation: clusters on nested shells
  nodeIds.forEach((id, i) => {
    const cid = communityMap[id] ?? 0
    const centroid = commCentroids[cid] || new THREE.Vector3(0,0,0)
    const dir = centroid.clone().normalize()
    // Distribute along the radial line to the centroid
    const offset = (i % 10) * 15
    posMap[id] = centroid.clone().add(dir.multiplyScalar(offset - 75))
  })
  return posMap
}

function makeLabel(text) {
  const div = document.createElement('div')
  div.textContent = text
  div.style.cssText = [
    'color:#ffd060', 'font-size:10px', 'font-family:monospace',
    'padding:2px 7px', 'background:rgba(0,0,0,0.8)',
    'border:1px solid rgba(255,208,96,0.35)', 'border-radius:3px',
    'pointer-events:none', 'white-space:nowrap', 'display:none',
  ].join(';')
  const obj = new CSS2DObject(div)
  obj.position.set(0, 7, 0)
  return { obj, div }
}

function randomOnSphere(r) {
  const u = Math.random(), v = Math.random()
  const theta = 2 * Math.PI * u, phi = Math.acos(2 * v - 1)
  return new THREE.Vector3(r * Math.sin(phi) * Math.cos(theta), r * Math.cos(phi), r * Math.sin(phi) * Math.sin(theta))
}

// ── Component ─────────────────────────────────────────────────────────────────
export function Brain3D({ events, connected, queryPaths, showLabels }) {
  const mountRef          = useRef(null)
  const stateRef          = useRef(null)
  const [graphSig, setGraphSig]         = useState(0)
  const [reloadMsg, setReloadMsg]       = useState('')
  const [selectedNode, setSelectedNode] = useState(null)
  const [layout, setLayout]             = useState(() => localStorage.getItem('cerebrum_layout') || 'clustered')
  const [colorScheme, setColorScheme]   = useState(() => localStorage.getItem('cerebrum_color_scheme') || 'classic')
  const [bloomStrength, setBloomStrength] = useState(() => parseFloat(localStorage.getItem('cerebrum_bloom') || '1.6'))
  const [focusedCid, setFocusedCid]       = useState(null)
  const [contextMenu, setContextMenu]     = useState(null) // { nodeId, cid, x, y }
  const [recomputeAlgo, setRecomputeAlgo] = useState('tsc')
  const [recomputing, setRecomputing]   = useState(false)

  // Persistence
  useEffect(() => { localStorage.setItem('cerebrum_layout', layout) }, [layout])
  useEffect(() => { localStorage.setItem('cerebrum_color_scheme', colorScheme) }, [colorScheme])
  useEffect(() => { localStorage.setItem('cerebrum_bloom', bloomStrength) }, [bloomStrength])
  const lastNodeCount    = useRef(null)
  const prevConnected    = useRef(connected)
  const queryFadeTimer   = useRef(null)
  const focusedCommunity = useRef(null)
  const forceRef         = useRef({ running: false, alpha: 1.0 })

  // ── Force simulation ──────────────────────────────────────────────────────
  const stepForce = (nodes, edges, posMap, alpha) => {
    const k = 40 // optimal distance
    const dt = 0.5 * alpha
    const forces = {}
    nodes.forEach(id => forces[id] = new THREE.Vector3())

    // Repulsion
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const u = nodes[i], v = nodes[j]
        const diff = posMap[u].clone().sub(posMap[v])
        const d2 = diff.lengthSq() || 1
        if (d2 > 600 * 600) continue
        const f = (k * k) / Math.sqrt(d2)
        const vec = diff.normalize().multiplyScalar(f * dt)
        forces[u].add(vec)
        forces[v].sub(vec)
      }
    }

    // Attraction
    edges.forEach(e => {
      const u = e.source_id ?? e.source, v = e.target_id ?? e.target
      if (!forces[u] || !forces[v]) return
      const diff = posMap[v].clone().sub(posMap[u])
      const d = diff.length() || 1
      const f = (d * d) / k
      const vec = diff.normalize().multiplyScalar(f * dt * 0.5)
      forces[u].add(vec)
      forces[v].sub(vec)
    })

    // Center gravity
    nodes.forEach(id => {
      const grav = posMap[id].clone().multiplyScalar(-0.02 * dt)
      forces[id].add(grav)
      posMap[id].add(forces[id].clampScalar(-50, 50))
    })
  }

  // WS reconnect — no graph reload (health poll handles node count changes)
  useEffect(() => {
    prevConnected.current = connected
  }, [connected])

  // Health poll every 30 s
  useEffect(() => {
    const id = setInterval(() => {
      api.health().then(r => {
        const n = r.data?.node_count ?? null
        // User requested to stop constant reloads. We only update the count ref now.
        lastNodeCount.current = n
      }).catch(() => {})
    }, 30_000)
    return () => clearInterval(id)
  }, [])

  // ── Scene setup ────────────────────────────────────────────────────────────
  useEffect(() => {
    const mount = mountRef.current
    if (!mount) return
    
    // Initialize state holder
    stateRef.current = {
      idToIndex: {}, indexToId: {}, baseColors: {}, posMap: {},
      nodeEdges: {}, labelMap: {}, commMap: {}, commMembers: {},
      commCentroids: {}, compactPosMap: {}, nodeIds: [], rawEdges: []
    }

    const W = mount.clientWidth || 800, H = mount.clientHeight || 600

    const scene  = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(55, W / H, 1, 10000)
    camera.position.set(0, 0, 1100)

    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setSize(W, H)
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.toneMapping = THREE.ACESFilmicToneMapping
    renderer.toneMappingExposure = 1.2
    mount.appendChild(renderer.domElement)

    const labelRenderer = new CSS2DRenderer()
    labelRenderer.setSize(W, H)
    labelRenderer.domElement.style.cssText = 'position:absolute;top:0;left:0;pointer-events:none;overflow:hidden'
    mount.appendChild(labelRenderer.domElement)

    // Lights
    scene.add(new THREE.AmbientLight(0xffffff, 0.4))
    const pointLight = new THREE.PointLight(0xffffff, 1.2)
    pointLight.position.set(200, 500, 800)
    scene.add(pointLight)

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true; controls.dampingFactor = 0.06
    controls.autoRotate = true;    controls.autoRotateSpeed = 0.2
    controls.minDistance = 80;     controls.maxDistance = 3000

    const bloomPass = new UnrealBloomPass(new THREE.Vector2(W, H), bloomStrength, 0.4, 0.0)
    const composer = new EffectComposer(renderer)
    composer.addPass(new RenderPass(scene, camera))
    composer.addPass(bloomPass)

    // Per-instance data
    const idToIndex  = {}  // node_id → instance index
    const indexToId  = {}  // instance index → node_id
    const baseColors = {}  // index → THREE.Color
    const posMap     = {}  // node_id → Vector3
    const nodeEdges  = {}  // node_id → [{dir, node, relation, weight}]
    const labelMap   = {}  // node_id → {obj, div}
    let   instMesh   = null
    let   edgeLines  = null
    let   commMap    = {}

    // Overlay lines for query path highlights (small count, individual Line objects)
    const queryOverlay = new THREE.Group()
    scene.add(queryOverlay)

    // Unified state holder initialization
    stateRef.current = {
      scene, camera, renderer, labelRenderer, composer, bloomPass, controls, queryOverlay,
      instMesh, edgeLines, idToIndex, indexToId, baseColors, posMap, nodeEdges, labelMap, commMap,
      commMembers: {}, commCentroids: {}, compactPosMap: {}, camAnim: null,
      nodeIds: [], rawEdges: [], streamingNodeCount: 0, streamingEdgeCount: 0,
      zoomOut, zoomIntoCommunity, applyLayout
    }

    Promise.all([api.edges(EDGE_LIMIT), api.communities()]).then(([edgesRes, commRes]) => {
      const s = stateRef.current
      if (!s) return // cleanup already ran

      s.commMap = commRes.data?.node_to_community || {}
      const rawEdges = edgesRes.data?.edges || []
      s.rawEdges = rawEdges

      // Build nodeEdges lookup
      for (const edge of rawEdges) {
        const src = edge.source_id ?? edge.source
        const tgt = edge.target_id ?? edge.target
        const rel = edge.relation_type ?? edge.relation ?? '—'
        const w   = edge.weight ?? null
        if (!s.nodeEdges[src]) s.nodeEdges[src] = []
        if (!s.nodeEdges[tgt]) s.nodeEdges[tgt] = []
        s.nodeEdges[src].push({ dir: 'out', node: tgt, relation: rel, weight: w })
        s.nodeEdges[tgt].push({ dir: 'in',  node: src, relation: rel, weight: w })
      }

      const nodeSet = new Set()
      for (const e of rawEdges) {
        const src = e.source_id ?? e.source, tgt = e.target_id ?? e.target
        if (src) nodeSet.add(src); if (tgt) nodeSet.add(tgt)
      }
      const nodeIds = [...nodeSet]
      s.nodeIds = nodeIds
      lastNodeCount.current = nodeIds.length

      // Community-clustered positions
      const positions = communityClusteredLayout(nodeIds, s.commMap)
      nodeIds.forEach((id, i) => { s.posMap[id] = positions[id] ?? randomOnSphere(300) })

      // Build commMembers + centroids + compact snapshot
      for (const id of nodeIds) {
        const cid = s.commMap[id] ?? 0
        if (!s.commMembers[cid]) s.commMembers[cid] = []
        s.commMembers[cid].push(id)
      }
      for (const [cid, members] of Object.entries(s.commMembers)) {
        const c = new THREE.Vector3()
        for (const id of members) c.add(s.posMap[id])
        c.divideScalar(members.length)
        s.commCentroids[cid] = c
      }
      for (const [id, pos] of Object.entries(s.posMap)) s.compactPosMap[id] = pos.clone()

      // ── InstancedMesh for nodes ──────────────────────────────────────────
      const sphereGeo = new THREE.IcosahedronGeometry(2.5, 2)
      const material  = new THREE.MeshStandardMaterial({
        roughness: 0.1, metalness: 0.8, emissiveIntensity: 0.5,
      })
      s.instMesh = new THREE.InstancedMesh(sphereGeo, material, MAX_NODES)
      s.instMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage)

      const mtx = new THREE.Matrix4()
      mtx.makeScale(0, 0, 0)
      for (let i = 0; i < MAX_NODES; i++) s.instMesh.setMatrixAt(i, mtx)

      nodeIds.forEach((id, i) => {
        mtx.makeScale(1, 1, 1); mtx.setPosition(s.posMap[id])
        s.instMesh.setMatrixAt(i, mtx)
        const c = getCommunityColor(s.commMap[id] ?? 0, colorScheme)
        s.instMesh.setColorAt(i, c)
        s.idToIndex[id] = i; s.indexToId[i] = id; s.baseColors[i] = c.clone()
      })
      s.instMesh.instanceMatrix.needsUpdate = true
      s.instMesh.instanceColor.needsUpdate  = true
      scene.add(s.instMesh)

      // ── LineSegments for edges ───────────────────────────────────────────
      const ePos = new Float32Array(MAX_EDGES * 2 * 3)
      const eCol = new Float32Array(MAX_EDGES * 2 * 3)
      
      let edgeIdx = 0
      for (const edge of rawEdges) {
        if (edgeIdx >= MAX_EDGES) break
        const src = edge.source_id ?? edge.source
        const tgt = edge.target_id ?? edge.target
        if (!s.posMap[src] || !s.posMap[tgt]) continue
        const p1 = s.posMap[src], p2 = s.posMap[tgt]
        const c1 = getCommunityColor(s.commMap[src] ?? 0, colorScheme)
        const c2 = getCommunityColor(s.commMap[tgt] ?? 0, colorScheme)
        
        const i6 = edgeIdx * 6
        ePos[i6+0] = p1.x; ePos[i6+1] = p1.y; ePos[i6+2] = p1.z
        ePos[i6+3] = p2.x; ePos[i6+4] = p2.y; ePos[i6+5] = p2.z
        eCol[i6+0] = c1.r * 0.5; eCol[i6+1] = c1.g * 0.5; eCol[i6+2] = c1.b * 0.5
        eCol[i6+3] = c2.r * 0.5; eCol[i6+4] = c2.g * 0.5; eCol[i6+5] = c2.b * 0.5
        edgeIdx++
      }

      const eGeo = new THREE.BufferGeometry()
      eGeo.setAttribute('position', new THREE.BufferAttribute(ePos, 3))
      eGeo.setAttribute('color',    new THREE.BufferAttribute(eCol, 3))
      s.edgeLines = new THREE.LineSegments(eGeo,
        new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.2 }))
      scene.add(s.edgeLines)

      s.streamingNodeCount = nodeIds.length
      s.streamingEdgeCount = edgeIdx

      // Fit camera to actual graph extents after load
      let maxR = 0
      for (const pos of Object.values(s.posMap)) maxR = Math.max(maxR, pos.length())
      const fitDist = Math.max(600, maxR / Math.tan((55 * Math.PI / 180) / 2) * 1.15)
      s.camAnim = {
        fromPos: camera.position.clone(), toPos: new THREE.Vector3(0, 0, fitDist),
        fromTarget: controls.target.clone(), toTarget: new THREE.Vector3(0, 0, 0),
        t: 0, duration: 90,
      }
    }).catch(console.error)

    // ── Raycaster ────────────────────────────────────────────────────────────
    const raycaster = new THREE.Raycaster()
    const mouse = new THREE.Vector2()
    const getHits = (e) => {
      if (!stateRef.current?.instMesh) return []
      const rect = renderer.domElement.getBoundingClientRect()
      mouse.x =  ((e.clientX - rect.left) / rect.width)  * 2 - 1
      mouse.y = -((e.clientY - rect.top)  / rect.height) * 2 + 1
      raycaster.setFromCamera(mouse, camera)
      return raycaster.intersectObject(stateRef.current.instMesh)
    }

    // Pulse a node white briefly
    const pulseNode = (nodeId) => {
      const idx = stateRef.current.idToIndex[nodeId]
      if (idx === undefined) return
      stateRef.current.instMesh.setColorAt(idx, new THREE.Color(1, 1, 1))
      stateRef.current.instMesh.instanceColor.needsUpdate = true
      setTimeout(() => {
        if (!stateRef.current?.instMesh) return
        stateRef.current.instMesh.setColorAt(idx, stateRef.current.baseColors[idx])
        stateRef.current.instMesh.instanceColor.needsUpdate = true
      }, 400)
    }

    // Zoom into a community cluster
    function zoomIntoCommunity(cid) {
      const s = stateRef.current
      const centroid = s.commCentroids?.[cid]
      if (!centroid) return
      const members = s.commMembers?.[cid] || []
      const expandR = Math.max(60, Math.sqrt(members.length) * 14)
      const expanded = fibonacciSphere(members.length, expandR)
      const mtx = new THREE.Matrix4()
      members.forEach((id, mi) => {
        const p = centroid.clone().add(expanded[mi])
        s.posMap[id] = p
        const idx = s.idToIndex[id]
        if (idx !== undefined) { mtx.setPosition(p); s.instMesh.setMatrixAt(idx, mtx) }
      })
      if (s.instMesh) s.instMesh.instanceMatrix.needsUpdate = true
      const dir = centroid.clone().normalize()
      const camDist = Math.max(180, expandR * 3.5)
      s.camAnim = {
        fromPos:    camera.position.clone(),
        toPos:      centroid.clone().add(dir.multiplyScalar(camDist)),
        fromTarget: controls.target.clone(),
        toTarget:   centroid.clone(),
        t: 0, duration: 70,
      }
      controls.autoRotate = false
      focusedCommunity.current = cid
      setFocusedCid(cid)
    }

    // Zoom back to full overview
    function zoomOut() {
      const s = stateRef.current
      if (!s?.compactPosMap) return
      const mtx = new THREE.Matrix4()
      for (const [id, pos] of Object.entries(s.compactPosMap)) {
        s.posMap[id] = pos.clone()
        const idx = s.idToIndex[id]
        if (idx !== undefined) { mtx.setPosition(pos); s.instMesh.setMatrixAt(idx, mtx) }
      }
      if (s.instMesh) s.instMesh.instanceMatrix.needsUpdate = true
      s.camAnim = {
        fromPos:    camera.position.clone(),
        toPos:      new THREE.Vector3(0, 0, 900),
        fromTarget: controls.target.clone(),
        toTarget:   new THREE.Vector3(0, 0, 0),
        t: 0, duration: 70,
      }
      controls.autoRotate = true
      focusedCommunity.current = null
      setFocusedCid(null)
    }

    function applyLayout(scheme, focalId = null) {
      const s = stateRef.current
      if (!s) return
      let newPositions = {}
      if (scheme === 'clustered') {
        newPositions = communityClusteredLayout(s.nodeIds, s.commMap)
      } else if (scheme === 'neighborhood') {
        newPositions = neighborhoodLayout(s.nodeIds, s.nodeEdges, focalId)
      } else if (scheme === 'cortical') {
        newPositions = corticalColumnLayout(s.nodeIds, s.commMap)
      } else if (scheme === 'hierarchical') {
        // Compute centroids first
        const commMembers = {}
        s.nodeIds.forEach(id => {
          const cid = s.commMap[id] ?? 0
          if (!commMembers[cid]) commMembers[cid] = []
          commMembers[cid].push(id)
        })
        const centroids = fibonacciSphere(Object.keys(commMembers).length, 400)
        const centroidMap = {}
        Object.keys(commMembers).forEach((cid, i) => centroidMap[cid] = centroids[i])
        newPositions = hierarchicalLayout(s.nodeIds, s.commMap, centroidMap)
      }

      if (Object.keys(newPositions).length > 0) {
        const mtx = new THREE.Matrix4()
        s.nodeIds.forEach((id, i) => {
          const p = newPositions[id] || s.posMap[id]
          s.posMap[id] = p.clone()
          const idx = s.idToIndex[id]
          if (idx !== undefined) { mtx.setPosition(p); s.instMesh.setMatrixAt(idx, mtx) }
          if (s.labelMap[id]) s.labelMap[id].helper.position.copy(p)
        })
        s.instMesh.instanceMatrix.needsUpdate = true
        // Update edges
        if (s.edgeLines) {
          const attr = s.edgeLines.geometry.attributes.position
          let idx = 0
          for (const edge of s.rawEdges) {
            const p1 = s.posMap[edge.source_id ?? edge.source]
            const p2 = s.posMap[edge.target_id ?? edge.target]
            if (p1 && p2) {
              attr.setXYZ(idx++, p1.x, p1.y, p1.z)
              attr.setXYZ(idx++, p2.x, p2.y, p2.z)
            }
          }
          attr.needsUpdate = true
        }

        // Broadcast to UE5/Other clients (Phase 103)
        api.broadcast('GUI_ADAPTATION', {
          action: 'layout_shift',
          target: scheme,
          focal_node: focalId
        }).catch(() => {})
      }
    }

    stateRef.current.applyLayout = applyLayout

    // Single click → open node details
    const onClick = (e) => {
      if (!stateRef.current?.instMesh) return
      const hits = getHits(e)
      if (!hits.length) {
        if (focusedCommunity.current !== null) zoomOut()
        setSelectedNode(null); setContextMenu(null); return
      }
      const nodeId = stateRef.current.indexToId[hits[0].instanceId]
      if (!nodeId) return
      const { nodeEdges: ne, commMap: cm } = stateRef.current
      const connections = ne[nodeId] || []
      setSelectedNode({
        id: nodeId, community: cm?.[nodeId] ?? '—',
        degree_out: connections.filter(c => c.dir === 'out').length,
        degree_in:  connections.filter(c => c.dir === 'in').length,
        outgoing:   connections.filter(c => c.dir === 'out'),
        incoming:   connections.filter(c => c.dir === 'in'),
      })
      pulseNode(nodeId)
    }

    // Double-click → zoom into community
    const onDblClick = (e) => {
      if (!stateRef.current?.instMesh) return
      const hits = getHits(e)
      if (!hits.length) { if (focusedCommunity.current !== null) zoomOut(); return }
      const nodeId = stateRef.current.indexToId[hits[0].instanceId]
      if (!nodeId) return
      const cid = stateRef.current.commMap?.[nodeId] ?? 0
      if (focusedCommunity.current === cid) { zoomOut(); return }
      zoomIntoCommunity(cid)
    }

    // Right-click → context menu
    const onContextMenu = (e) => {
      e.preventDefault()
      if (!stateRef.current?.instMesh) return
      const hits = getHits(e)
      if (!hits.length) { setContextMenu(null); return }
      const nodeId = stateRef.current.indexToId[hits[0].instanceId]
      if (!nodeId) return
      const cid = stateRef.current.commMap?.[nodeId] ?? 0
      setContextMenu({ nodeId, cid, x: e.clientX, y: e.clientY })
    }

    // ESC → zoom out
    const onKey = (e) => {
      if (e.key === 'Escape') {
        if (focusedCommunity.current !== null) zoomOut()
        setContextMenu(null)
      }
    }

    renderer.domElement.addEventListener('click', onClick)
    renderer.domElement.addEventListener('dblclick', onDblClick)
    renderer.domElement.addEventListener('contextmenu', onContextMenu)
    window.addEventListener('keydown', onKey)

    // ── Render loop ──────────────────────────────────────────────────────────
    let frameId
    const animate = () => {
      frameId = requestAnimationFrame(animate)
      const s = stateRef.current
      if (!s) return

      // Force simulation step
      if (forceRef.current.running && s.nodeIds && s.rawEdges) {
        stepForce(s.nodeIds, s.rawEdges, s.posMap, forceRef.current.alpha)
        forceRef.current.alpha *= 0.992 // cooling
        if (forceRef.current.alpha < 0.005) forceRef.current.running = false

        // Update InstancedMesh matrices
        const mtx = new THREE.Matrix4()
        s.nodeIds.forEach((id, i) => {
          mtx.setPosition(s.posMap[id])
          s.instMesh.setMatrixAt(i, mtx)
          if (s.labelMap[id]) s.labelMap[id].helper.position.copy(s.posMap[id])
        })
        s.instMesh.instanceMatrix.needsUpdate = true

        // Update edge positions
        if (s.edgeLines) {
          const attr = s.edgeLines.geometry.attributes.position
          let idx = 0
          for (const edge of s.rawEdges) {
            const p1 = s.posMap[edge.source_id ?? edge.source]
            const p2 = s.posMap[edge.target_id ?? edge.target]
            if (p1 && p2) {
              attr.setXYZ(idx++, p1.x, p1.y, p1.z)
              attr.setXYZ(idx++, p2.x, p2.y, p2.z)
            }
          }
          attr.needsUpdate = true
        }
      }

      controls.update()
      const anim = s.camAnim
      if (anim) {
        anim.t = Math.min(anim.t + 1 / anim.duration, 1)
        const ease = 1 - Math.pow(1 - anim.t, 3)
        camera.position.lerpVectors(anim.fromPos, anim.toPos, ease)
        controls.target.lerpVectors(anim.fromTarget, anim.toTarget, ease)
        if (anim.t >= 1) s.camAnim = null
      }
      composer.render()
      labelRenderer.render(scene, camera)
    }
    animate()

    const ro = new ResizeObserver(() => {
      const W = mount.clientWidth, H = mount.clientHeight
      camera.aspect = W / H; camera.updateProjectionMatrix()
      renderer.setSize(W, H); composer.setSize(W, H); labelRenderer.setSize(W, H)
    })
    ro.observe(mount)

    return () => {
      cancelAnimationFrame(frameId)
      ro.disconnect()
      renderer.domElement.removeEventListener('click', onClick)
      renderer.domElement.removeEventListener('dblclick', onDblClick)
      renderer.domElement.removeEventListener('contextmenu', onContextMenu)
      window.removeEventListener('keydown', onKey)
      controls.dispose(); renderer.dispose()
      if (mount.contains(renderer.domElement))      mount.removeChild(renderer.domElement)
      if (mount.contains(labelRenderer.domElement)) mount.removeChild(labelRenderer.domElement)
      stateRef.current = null
    }
  }, [graphSig, colorScheme]) // eslint-disable-line react-hooks/exhaustive-deps

  // ── Bloom strength ────────────────────────────────────────────────────────
  useEffect(() => {
    if (stateRef.current?.bloomPass) stateRef.current.bloomPass.strength = bloomStrength
  }, [bloomStrength])

  // ── Global label toggle ────────────────────────────────────────────────────
  useEffect(() => {
    if (!stateRef.current?.posMap) return
    const { posMap, labelMap, scene } = stateRef.current

    if (showLabels) {
      // Lazily create labels for every loaded node
      for (const [id, pos] of Object.entries(posMap)) {
        if (!labelMap[id]) {
          const { obj, div } = makeLabel(id.replace(/_/g, ' '))
          const helper = new THREE.Object3D()
          helper.position.copy(pos)
          helper.add(obj)
          scene.add(helper)
          labelMap[id] = { obj, div, helper }
        }
        labelMap[id].div.style.display = 'block'
        labelMap[id].div.style.color = '#c9d1d9'
        labelMap[id].div.style.borderColor = 'rgba(255,255,255,0.15)'
      }
    } else {
      // Hide all (query-highlighted ones re-show on next query effect)
      for (const entry of Object.values(labelMap)) {
        entry.div.style.display = 'none'
      }
    }
  }, [showLabels])

  // ── Query path highlighting ────────────────────────────────────────────────
  useEffect(() => {
    if (!queryPaths?.length || !stateRef.current?.instMesh) return
    if (queryFadeTimer.current) clearTimeout(queryFadeTimer.current)

    const { instMesh: im, edgeLines: el, idToIndex, baseColors, posMap,
            commMap: cm, queryOverlay, labelMap, scene } = stateRef.current

    const highlightIds  = new Set()
    const highlightEdges = []  // [{src, tgt}]
    for (const p of queryPaths.slice(0, 3)) {
      const entities = (p.path || []).filter(n => n.type === 'entity').map(n => n.id)
      entities.forEach(id => highlightIds.add(id))
      for (let i = 0; i < entities.length - 1; i++)
        highlightEdges.push({ src: entities[i], tgt: entities[i + 1] })
    }

    // Dim all nodes, gold up path nodes
    const n = Object.keys(idToIndex).length
    for (let i = 0; i < n; i++) im.setColorAt(i, COLOR_DIM)
    for (const id of highlightIds) {
      const idx = idToIndex[id]
      if (idx !== undefined) im.setColorAt(idx, COLOR_QUERY_NODE)
    }
    im.instanceColor.needsUpdate = true

    // Dim edge lines
    if (el) el.material.opacity = 0.03

    // Overlay bright cyan lines for path edges
    while (queryOverlay.children.length) queryOverlay.remove(queryOverlay.children[0])
    for (const { src, tgt } of highlightEdges) {
      if (!posMap[src] || !posMap[tgt]) continue
      const geo = new THREE.BufferGeometry().setFromPoints([posMap[src], posMap[tgt]])
      const line = new THREE.Line(geo,
        new THREE.LineBasicMaterial({ color: COLOR_QUERY_EDGE, transparent: true, opacity: 1.0 }))
      queryOverlay.add(line)
    }

    // Show labels for highlighted nodes
    for (const id of highlightIds) {
      if (!labelMap[id]) {
        const { obj, div } = makeLabel(id.replace(/_/g, ' '))
        const helper = new THREE.Object3D()
        helper.position.copy(posMap[id])
        helper.add(obj)
        stateRef.current.scene.add(helper)
        labelMap[id] = { obj, div, helper }
      }
      labelMap[id].div.style.display = 'block'
    }

    queryFadeTimer.current = setTimeout(() => {
      if (!stateRef.current?.instMesh) return
      const { instMesh: im2, edgeLines: el2, baseColors: bc, labelMap: lm, queryOverlay: qo } = stateRef.current
      for (const [i, c] of Object.entries(bc)) im2.setColorAt(Number(i), c)
      im2.instanceColor.needsUpdate = true
      if (el2) el2.material.opacity = 0.2
      while (qo.children.length) qo.remove(qo.children[0])
      for (const entry of Object.values(lm)) entry.div.style.display = 'none'
    }, 6000)
  }, [queryPaths])

  // ── Telemetry events ───────────────────────────────────────────────────────
  useEffect(() => {
    if (!events.length || !stateRef.current?.instMesh) return
    const latest = events[0]
    const { instMesh: im, baseColors, idToIndex, posMap, queryOverlay, commMap } = stateRef.current

    const flashIdx = (id, color, ms) => {
      const idx = idToIndex[id]
      if (idx === undefined) return
      im.setColorAt(idx, color)
      im.instanceColor.needsUpdate = true
      setTimeout(() => {
        if (!stateRef.current?.instMesh) return
        im.setColorAt(idx, baseColors[idx])
        im.instanceColor.needsUpdate = true
      }, ms)
    }

    if (latest.event_type === 'SYNAPTIC_PULSE') {
      const { source_node, target_node, is_SynapticBridge } = latest.payload || {}
      const color = is_SynapticBridge ? COLOR_WORMHOLE : new THREE.Color(1, 1, 1)
      flashIdx(source_node, color, is_SynapticBridge ? 500 : 300)
      flashIdx(target_node, color, is_SynapticBridge ? 500 : 300)

      if (is_SynapticBridge && posMap[source_node] && posMap[target_node]) {
        const geo  = new THREE.BufferGeometry().setFromPoints([posMap[source_node], posMap[target_node]])
        const line = new THREE.Line(geo, new THREE.LineBasicMaterial({ color: COLOR_WORMHOLE, transparent: true, opacity: 1.0 }))
        queryOverlay.add(line)
        setTimeout(() => queryOverlay.remove(line), 700)
      }
    }

    if (latest.event_type === 'SYNAPTOGENESIS') {
      const { source_node, target_node } = latest.payload || {}
      if (!posMap[source_node] || !posMap[target_node]) return
      
      // Permanent addition to edgeLines geometry
      if (stateRef.current.edgeLines && stateRef.current.streamingEdgeCount < MAX_EDGES) {
        const s = stateRef.current
        const idx = s.streamingEdgeCount
        const attr = s.edgeLines.geometry.attributes.position
        const cattr = s.edgeLines.geometry.attributes.color
        const p1 = posMap[source_node], p2 = posMap[target_node]
        const c1 = getCommunityColor(s.commMap[source_node] ?? 0, colorScheme)
        const c2 = getCommunityColor(s.commMap[target_node] ?? 0, colorScheme)
        
        const i6 = idx * 6
        attr.setXYZ(idx * 2, p1.x, p1.y, p1.z)
        attr.setXYZ(idx * 2 + 1, p2.x, p2.y, p2.z)
        cattr.setXYZ(idx * 2, c1.r * 0.5, c1.g * 0.5, c1.b * 0.5)
        cattr.setXYZ(idx * 2 + 1, c2.r * 0.5, c2.g * 0.5, c2.b * 0.5)
        
        attr.needsUpdate = true
        cattr.needsUpdate = true
        s.streamingEdgeCount++
        s.rawEdges.push({ source: source_node, target: target_node })
      }

      // Visual pulse
      const geo  = new THREE.BufferGeometry().setFromPoints([posMap[source_node], posMap[target_node]])
      const mat  = new THREE.LineBasicMaterial({ color: COLOR_NEW_EDGE, transparent: true, opacity: 1.0 })
      const line = new THREE.Line(geo, mat)
      queryOverlay.add(line)
      let pulses = 0
      const pulse = setInterval(() => {
        mat.opacity = ++pulses % 2 === 0 ? 1.0 : 0.15
        if (pulses >= 6) { clearInterval(pulse); queryOverlay.remove(line) }
      }, 250)
    }

    if (latest.event_type === 'NEUROGENESIS') {
      const { node_id, community_id } = latest.payload || {}
      if (!idToIndex[node_id] && stateRef.current.streamingNodeCount < MAX_NODES) {
        const s = stateRef.current
        const idx = s.streamingNodeCount
        const pos = randomOnSphere(350)
        posMap[node_id] = pos
        idToIndex[node_id] = idx
        indexToId[idx] = node_id
        s.nodeIds.push(node_id)
        s.commMap[node_id] = community_id ?? 0
        
        const c = getCommunityColor(s.commMap[node_id], colorScheme)
        s.baseColors[idx] = c.clone()
        
        const mtx = new THREE.Matrix4()
        mtx.setPosition(pos)
        im.setMatrixAt(idx, mtx)
        im.setColorAt(idx, c)
        
        im.instanceMatrix.needsUpdate = true
        im.instanceColor.needsUpdate = true
        s.streamingNodeCount++
      }
      flashIdx(node_id, new THREE.Color(1, 1, 1), 800)
    }
  }, [events, colorScheme])

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      <div ref={mountRef} style={{ width: '100%', height: '100%' }} />
      <GraphToolbar
        stateRef={stateRef}
        layout={layout}
        setLayout={setLayout}
        colorScheme={colorScheme}
        setColorScheme={setColorScheme}
        forceRef={forceRef}
        onReload={() => setGraphSig(s => s + 1)}
        onToast={(msg) => showToast(setReloadMsg, msg)}
      />
      <div style={bloomControlStyle}>
        <span style={{ color: '#aaa', fontSize: 11, marginRight: 6 }}>Bloom</span>
        <input
          type="range" min="0" max="3" step="0.05"
          value={bloomStrength}
          onChange={e => setBloomStrength(parseFloat(e.target.value))}
          style={{ width: 90, accentColor: '#7af' }}
        />
        <span style={{ color: '#7af', fontSize: 11, marginLeft: 6, minWidth: 28 }}>{bloomStrength.toFixed(1)}</span>
      </div>
      {focusedCid !== null && (
        <button style={backBtnStyle} onClick={() => stateRef.current?.zoomOut?.()}>
          ← Overview
        </button>
      )}
      {contextMenu && (
        <ContextMenu
          menu={contextMenu}
          onZoomCommunity={() => {
            const s = stateRef.current; if (!s) return
            if (focusedCommunity.current === contextMenu.cid) s.zoomOut?.()
            else s.zoomIntoCommunity?.(contextMenu.cid)
            setContextMenu(null)
          }}
          onNeighborhood={() => {
            const s = stateRef.current; if (!s) return
            setLayout('neighborhood')
            s.applyLayout?.('neighborhood', contextMenu.nodeId)
            setContextMenu(null)
            onToast?.("Focused on neighborhood of " + contextMenu.nodeId)
          }}
          onHierarchical={() => {
            setLayout('hierarchical')
            stateRef.current?.applyLayout?.('hierarchical')
            setContextMenu(null)
          }}
          onDetails={() => {
            const s = stateRef.current; if (!s) return
            const { nodeEdges: ne, commMap: cm } = s
            const connections = ne[contextMenu.nodeId] || []
            setSelectedNode({
              id: contextMenu.nodeId, community: cm?.[contextMenu.nodeId] ?? '—',
              degree_out: connections.filter(c => c.dir === 'out').length,
              degree_in:  connections.filter(c => c.dir === 'in').length,
              outgoing:   connections.filter(c => c.dir === 'out'),
              incoming:   connections.filter(c => c.dir === 'in'),
            })
            setContextMenu(null)
          }}
          onClose={() => setContextMenu(null)}
        />
      )}
      {reloadMsg && <div style={toastStyle}>{reloadMsg}</div>}
      {selectedNode && <NodePanel node={selectedNode} onClose={() => setSelectedNode(null)} />}
    </div>
  )
}

// ── Node metadata panel ────────────────────────────────────────────────────────
function NodePanel({ node, onClose }) {
  return (
    <div style={panelStyle}>
      <div style={panelHeader}>
        <span style={panelTitle}>{node.id.replace(/_/g, ' ')}</span>
        <button onClick={onClose} style={closeBtn}>✕</button>
      </div>
      <Row label="Community"  value={`#${node.community}`} />
      <Row label="Out-degree" value={node.degree_out} />
      <Row label="In-degree"  value={node.degree_in} />
      {node.outgoing.length > 0 && (
        <Section title="Outgoing">
          {node.outgoing.slice(0, 20).map((e, i) => (
            <EdgeRow key={i} dir="→" rel={e.relation} node={e.node} weight={e.weight} />
          ))}
          {node.outgoing.length > 20 && <More n={node.outgoing.length - 20} />}
        </Section>
      )}
      {node.incoming.length > 0 && (
        <Section title="Incoming">
          {node.incoming.slice(0, 20).map((e, i) => (
            <EdgeRow key={i} dir="←" rel={e.relation} node={e.node} weight={e.weight} />
          ))}
          {node.incoming.length > 20 && <More n={node.incoming.length - 20} />}
        </Section>
      )}
    </div>
  )
}

function Row({ label, value }) {
  return (
    <div style={rowStyle}>
      <span style={rowLabel}>{label}</span>
      <span style={rowValue}>{value}</span>
    </div>
  )
}
function Section({ title, children }) {
  return (
    <div style={{ marginTop: 10 }}>
      <div style={sectionTitle}>{title}</div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>{children}</div>
    </div>
  )
}
function EdgeRow({ dir, rel, node, weight }) {
  return (
    <div style={edgeRowStyle}>
      <span style={{ color: '#8b949e', fontSize: 10, minWidth: 10 }}>{dir}</span>
      <span style={{ color: '#58a6ff', fontSize: 10, fontFamily: 'monospace', minWidth: 80 }}>{rel}</span>
      <span style={{ color: '#c9d1d9', fontSize: 10, flex: 1 }}>{node.replace(/_/g, ' ')}</span>
      {weight != null && <span style={{ color: '#8b949e', fontSize: 9 }}>{Number(weight).toFixed(2)}</span>}
    </div>
  )
}
function More({ n }) {
  return <div style={{ color: '#8b949e', fontSize: 10, padding: '2px 4px' }}>+ {n} more…</div>
}

// ── Graph toolbar ──────────────────────────────────────────────────────────────
function GraphToolbar({ stateRef, layout, setLayout, colorScheme, setColorScheme, forceRef, onReload, onToast }) {
  const [loopRunning, setLoopRunning] = useState(false)
  const [autoRotate, setAutoRotate]   = useState(true)
  const [busy, setBusy]               = useState({})
  const [recomputeAlgo, setRecomputeAlgo] = useState('tsc')

  const run = async (key, fn, successMsg) => {
    setBusy(b => ({ ...b, [key]: true }))
    try {
      await fn()
      if (successMsg) onToast(successMsg)
    } catch (e) {
      onToast(`Error: ${e?.response?.data?.detail || e.message || key}`)
    } finally {
      setBusy(b => ({ ...b, [key]: false }))
    }
  }

  const triggerRecompute = () => {
    run('recompute', async () => {
      const r = await api.recomputeCommunities(recomputeAlgo)
      const s = stateRef.current
      if (s) {
        s.commMap = r.data.node_to_community
        s.applyLayout?.('clustered')
      }
    }, `Re-organized via ${recomputeAlgo.toUpperCase()}`)
  }

  // Sync loop status on mount
  useState(() => {
    api.loopStatus().then(r => setLoopRunning(r.data?.running ?? false)).catch(() => {})
  })

  const toggleRotate = () => {
    const s = stateRef.current
    if (!s?.controls) return
    const next = !s.controls.autoRotate
    s.controls.autoRotate = next
    setAutoRotate(next)
  }

  const resetCamera = () => {
    const s = stateRef.current
    if (!s) return
    s.zoomOut?.()
  }

  const onLayoutChange = (e) => {
    const next = e.target.value
    setLayout(next)
    const s = stateRef.current
    if (next === 'force') {
      forceRef.current.running = true
      forceRef.current.alpha = 1.0
      if (s?.controls) {
        s.controls.autoRotate = false
        setAutoRotate(false)
      }
    } else {
      forceRef.current.running = false
      s?.applyLayout?.(next)
      if (next === 'clustered') resetCamera()
    }
  }

  const toggleLoop = () => {
    if (loopRunning) {
      run('loop', () => api.loopStop().then(() => setLoopRunning(false)), 'Research loop stopped')
    } else {
      run('loop', () => api.loopStart().then(() => setLoopRunning(true)), 'Research loop started')
    }
  }

  const Btn = ({ id, label, icon, onClick, color, title, active }) => (
    <button
      title={title || label}
      onClick={onClick}
      disabled={!!busy[id]}
      style={{
        ...tbBtn,
        ...(color ? { color, borderColor: color + '55' } : {}),
        ...(active ? { background: 'rgba(0,210,255,0.15)', borderColor: '#00d2ff' } : {})
      }}
    >
      <span style={tbIcon}>{icon}</span>
      <span>{busy[id] ? '…' : label}</span>
    </button>
  )

  return (
    <div style={tbStyle}>
      <div style={tbGroup}>
        <div style={tbGroupLabel}>View</div>
        <Btn id="cam"    label="Reset Camera"   icon="⊙" onClick={resetCamera} title="Zoom to full graph (ESC)" />
        <Btn id="rot"    label={autoRotate ? 'Stop Rotate' : 'Auto Rotate'} icon="↻" onClick={toggleRotate} />
        <div style={{ ...tbBtn, border: 'none', background: 'none' }}>
          <select style={selectStyle} value={layout} onChange={onLayoutChange}>
            <option value="clustered">Community Clusters</option>
            <option value="cortical">Cortical Columns (3D)</option>
            <option value="hierarchical">Hierarchical Rings</option>
            <option value="neighborhood">Neural Neighborhood</option>
            <option value="force">Force-Directed</option>
          </select>
        </div>
        <div style={{ ...tbBtn, border: 'none', background: 'none' }}>
          <select
            style={selectStyle}
            value={colorScheme}
            onChange={e => {
              const next = e.target.value
              setColorScheme(next)
              const s = stateRef.current
              if (s?.instMesh) {
                s.nodeIds.forEach((id, i) => {
                  const c = getCommunityColor(s.commMap[id] ?? 0, next)
                  s.instMesh.setColorAt(i, c)
                  s.baseColors[i] = c.clone()
                })
                s.instMesh.instanceColor.needsUpdate = true
              }
            }}
          >
            <option value="classic">Classic</option>
            <option value="neon">Neon</option>
            <option value="monochrome">Terminal</option>
            <option value="cyber">Cyberpunk</option>
          </select>
        </div>
        <Btn id="reload" label="Reload Graph"   icon="⟳" onClick={() => { onReload(); onToast('Graph reloading…') }} />
      </div>
      <div style={tbDivider} />
      <div style={tbGroup}>
        <div style={tbGroupLabel}>Graph Ops</div>
        <Btn id="rem_all"  label="Run REM"        icon="✦" color="#c084fc"
          onClick={() => run('rem_all', () => api.rem(), 'REM complete')}
          title="Prune + Consolidate + Synthesize" />
        <Btn id="retrain"  label="Retrain Params" icon="⬆" color="#4ade80"
          onClick={() => run('retrain', () => api.retrain(), 'Parameters retrained')} />
      </div>
      <div style={tbDivider} />
      <div style={tbGroup}>
        <div style={tbGroupLabel}>Re-Organize</div>
        <div style={{ ...tbBtn, border: 'none', background: 'none' }}>
          <select style={selectStyle} value={recomputeAlgo} onChange={e => setRecomputeAlgo(e.target.value)}>
            <option value="tsc">TSC (Triple-Signal)</option>
            <option value="dscf">DSCF (Dual-Signal)</option>
            <option value="louvain">Louvain (Modularity)</option>
            <option value="lpa">LPA (Local)</option>
          </select>
        </div>
        <Btn id="recompute" label="Recompute" icon="⚙" onClick={triggerRecompute} />
      </div>
      <div style={tbDivider} />
      <div style={tbGroup}>
        <div style={tbGroupLabel}>Research</div>
        <Btn id="loop" label={loopRunning ? 'Stop Loop' : 'Start Loop'} icon={loopRunning ? '■' : '▶'}
          color={loopRunning ? '#f85149' : '#3fb950'}
          onClick={toggleLoop}
          title="Autonomous discovery loop" />
        <Btn id="rollback" label="Rollback Last" icon="↩" color="#fbbf24"
          onClick={() => run('rollback', async () => {
            const r = await api.provenanceBatches(1)
            const batch = r.data?.batches?.[0]
            if (!batch) throw new Error('No batches to roll back')
            await api.rollbackBatch(batch.batch_id)
            onReload()
          }, 'Last batch rolled back')} />
      </div>
    </div>
  )
}

function ContextMenu({ menu, onZoomCommunity, onNeighborhood, onHierarchical, onDetails, onClose }) {
  return (
    <div style={{ ...ctxMenuStyle, left: menu.x, top: menu.y }} onMouseLeave={onClose}>
      <div style={ctxTitle}>{menu.nodeId.replace(/_/g, ' ')}</div>
      <div style={ctxDivider} />
      <button style={ctxItem} onClick={onZoomCommunity}>
        {'\u2609'} Zoom to Community #{menu.cid}
      </button>
      <button style={ctxItem} onClick={onNeighborhood}>
        {'\u2608'} Focus Neighborhood
      </button>
      <button style={ctxItem} onClick={onHierarchical}>
        {'\u2631'} Use as Hierarchical Anchor
      </button>
      <button style={ctxItem} onClick={onDetails}>
        {'\u2139'} View Node Details
      </button>
      <div style={ctxDivider} />
      <button style={{ ...ctxItem, color: '#8b949e' }} onClick={onClose}>Cancel</button>
    </div>
  )
}

function showToast(setter, msg) {
  setter(msg); setTimeout(() => setter(''), 2500)
}

// ── Styles ────────────────────────────────────────────────────────────────────
const panelStyle = {
  position: 'absolute', top: 52, left: 12, width: 290,
  maxHeight: 'calc(100vh - 80px)',
  background: 'rgba(0,0,0,0.85)', backdropFilter: 'blur(12px)',
  border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8,
  padding: '12px 14px', overflowY: 'auto', zIndex: 20,
}
const panelHeader  = { display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }
const panelTitle   = { fontSize: 13, fontWeight: 700, color: '#e6edf3', letterSpacing: 0.5 }
const closeBtn     = { background: 'none', border: 'none', color: '#8b949e', fontSize: 14, cursor: 'pointer' }
const rowStyle     = { display: 'flex', justifyContent: 'space-between', padding: '3px 0', borderBottom: '1px solid rgba(255,255,255,0.05)' }
const rowLabel     = { fontSize: 11, color: '#8b949e' }
const rowValue     = { fontSize: 11, color: '#c9d1d9', fontFamily: 'monospace' }
const sectionTitle = { fontSize: 10, fontWeight: 700, color: '#58a6ff', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 5 }
const edgeRowStyle = { display: 'flex', gap: 5, alignItems: 'center', padding: '2px 4px', borderRadius: 3, background: 'rgba(255,255,255,0.03)' }
const bloomControlStyle = {
  position: 'absolute', bottom: 16, right: 16,
  display: 'flex', alignItems: 'center',
  background: 'rgba(0,0,0,0.55)', border: '1px solid rgba(255,255,255,0.1)',
  borderRadius: 6, padding: '5px 10px',
}
const backBtnStyle = {
  position: 'absolute', top: 52, right: 16, zIndex: 20,
  background: 'rgba(0,0,0,0.7)', border: '1px solid rgba(122,175,255,0.4)',
  color: '#7af', fontSize: 12, cursor: 'pointer', borderRadius: 5, padding: '5px 14px',
}
const ctxMenuStyle = {
  position: 'fixed', zIndex: 50, minWidth: 200,
  background: 'rgba(13,17,23,0.97)', border: '1px solid rgba(255,255,255,0.12)',
  borderRadius: 8, padding: '6px 0', boxShadow: '0 8px 24px rgba(0,0,0,0.6)',
}
const ctxTitle   = { fontSize: 11, color: '#8b949e', padding: '4px 14px 6px', fontFamily: 'monospace', maxWidth: 180, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }
const ctxDivider = { height: 1, background: 'rgba(255,255,255,0.08)', margin: '3px 0' }
const ctxItem    = { display: 'block', width: '100%', textAlign: 'left', background: 'none', border: 'none', color: '#c9d1d9', fontSize: 12, padding: '6px 14px', cursor: 'pointer' }
const toastStyle   = {
  position: 'absolute', bottom: 16, left: '50%', transform: 'translateX(-50%)',
  background: 'rgba(0,210,255,0.15)', border: '1px solid rgba(0,210,255,0.4)',
  color: '#00d2ff', fontSize: 11, letterSpacing: 1.5, textTransform: 'uppercase',
  padding: '5px 16px', borderRadius: 4, pointerEvents: 'none',
}
const selectStyle = {
  background: '#0d1117', border: '1px solid rgba(255,255,255,0.15)',
  borderRadius: 4, color: '#c9d1d9', fontSize: 11, padding: '2px 4px',
  outline: 'none', cursor: 'pointer',
}
const tbStyle   = { position: 'absolute', top: 52, left: '50%', transform: 'translateX(-50%)', display: 'flex', gap: 6, background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(10px)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 8, padding: '5px 8px', zIndex: 10 }
const tbGroup   = { display: 'flex', gap: 4, alignItems: 'center' }
const tbGroupLabel = { fontSize: 9, fontWeight: 700, color: 'rgba(255,255,255,0.25)', textTransform: 'uppercase', letterSpacing: 1, marginRight: 4 }
const tbDivider = { width: 1, height: 20, background: 'rgba(255,255,255,0.12)', margin: '0 4px' }
const tbBtn     = { display: 'flex', alignItems: 'center', gap: 5, background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.12)', borderRadius: 5, color: 'rgba(255,255,255,0.6)', fontSize: 11, padding: '4px 9px', cursor: 'pointer', transition: 'all 0.1s' }
const tbIcon    = { fontSize: 13, color: 'rgba(255,255,255,0.4)' }
