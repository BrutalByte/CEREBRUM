import axios from 'axios'

const BASE = '/v1'

export const api = {
  query: (q, maxHops = 3, beamWidth = 5) =>
    axios.post(`${BASE}/query`, { query: q, max_hop: maxHops, beam_width: beamWidth }),

  health: () => axios.get(`${BASE}/health`),

  communities: () => axios.get(`${BASE}/communities`),

  edges: (limit = 500) => axios.get(`${BASE}/graph/edges?limit=${limit}`),

  chemical: () => axios.get(`${BASE}/chemical`),

  loopStatus: () => axios.get(`${BASE}/research/loop/status`),

  loopStart: () => axios.post(`${BASE}/research/loop/start`),

  loopStop: () => axios.post(`${BASE}/research/loop/stop`),

  provenanceStats: () => axios.get(`${BASE}/research/provenance/stats`),

  provenanceBatches: (n = 20) => axios.get(`${BASE}/research/provenance/batches?n=${n}`),

  rollbackBatch: (batchId) => axios.post(`${BASE}/research/provenance/rollback/${batchId}`),

  rem: (ops = ['prune', 'consolidate', 'synthesize']) =>
    axios.post(`${BASE}/rem/run`, { operations: ops }),

  retrain: () => axios.post(`${BASE}/retrain`),

  params: () => axios.get(`${BASE}/params`),

  goals: () => axios.get(`${BASE}/goals`),

  addGoal: (text, priority = 1) =>
    axios.post(`${BASE}/goals`, { text, priority }),

  deleteGoal: (id) => axios.delete(`${BASE}/goals/${id}`),

  goalHistory: (id) => axios.get(`${BASE}/goals/${id}/history`),

  loopConfigure: (config) => axios.post(`${BASE}/research/loop/configure`, config),

  broadcast: (eventType, payload) =>
    axios.post(`${BASE}/telemetry/broadcast`, { event_type: eventType, payload }),

  recomputeCommunities: (algorithm = 'tsc', resolution = 1.0) =>
    axios.post(`${BASE}/communities/recompute`, { algorithm, resolution }),
}
