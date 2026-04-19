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
}
