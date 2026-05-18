/**
 * CEREBRUM TypeScript SDK — typed wrapper over the REST API.
 *
 *   import { Cerebrum } from "./cerebrum";
 *
 *   const c = new Cerebrum("http://localhost:8200");
 *   const result = await c.ask("Who directed Inception?");
 *   console.log(result.answer, result.trace_path);
 */

export interface TraceStep {
  entity: string;
  relation: string;
}

export interface TopCandidate {
  entity: string;
  confidence: number;
}

export interface Result {
  answer: string;
  confidence: number;
  trace_path: TraceStep[];
  top_k: TopCandidate[];
  elapsed_ms: number;
  query: string;
}

export interface KBStats {
  entities: number;
  relations: number;
  relation_types: number;
  communities: number;
}

export interface CerebrumOptions {
  /** Base URL of the CEREBRUM REST API (default: http://localhost:8200) */
  baseUrl?: string;
  /** Default beam width for traversal (default: 10) */
  beamWidth?: number;
  /** Default max hops (default: 3) */
  maxHop?: number;
  /** Default top-k results (default: 5) */
  topK?: number;
  /** Request timeout in milliseconds (default: 30000) */
  timeoutMs?: number;
}

export class CerebrumError extends Error {
  constructor(
    message: string,
    public readonly statusCode?: number,
    public readonly detail?: unknown,
  ) {
    super(message);
    this.name = "CerebrumError";
  }
}

export class Cerebrum {
  private readonly baseUrl: string;
  private readonly beamWidth: number;
  private readonly maxHop: number;
  private readonly topK: number;
  private readonly timeoutMs: number;

  constructor(baseUrlOrOptions?: string | CerebrumOptions) {
    const opts: CerebrumOptions =
      typeof baseUrlOrOptions === "string"
        ? { baseUrl: baseUrlOrOptions }
        : (baseUrlOrOptions ?? {});

    this.baseUrl = (opts.baseUrl ?? "http://localhost:8200").replace(/\/$/, "");
    this.beamWidth = opts.beamWidth ?? 10;
    this.maxHop = opts.maxHop ?? 3;
    this.topK = opts.topK ?? 5;
    this.timeoutMs = opts.timeoutMs ?? 30_000;
  }

  /**
   * Answer a natural-language question by traversing the knowledge graph.
   */
  async ask(
    question: string,
    opts?: { beamWidth?: number; maxHop?: number; topK?: number },
  ): Promise<Result> {
    const t0 = Date.now();
    const payload = {
      query: question,
      beam_width: opts?.beamWidth ?? this.beamWidth,
      max_hop: opts?.maxHop ?? this.maxHop,
      top_k: opts?.topK ?? this.topK,
      mode: "consensus",
    };

    const raw = await this._post("/v1/query", payload);
    return this._toResult(raw, question, Date.now() - t0);
  }

  /**
   * Query by entity ID or label directly (no NL parsing step).
   */
  async query(
    entity: string,
    opts?: { beamWidth?: number; maxHop?: number; topK?: number },
  ): Promise<Result> {
    return this.ask(entity, opts);
  }

  /**
   * Retrieve the full reasoning trace for the last query.
   */
  async trace(query: string): Promise<TraceStep[]> {
    const raw = await this._get(`/v1/trace?query=${encodeURIComponent(query)}`);
    if (!Array.isArray(raw?.path)) return [];
    return raw.path.map((step: any) => ({
      entity: String(step.entity ?? step.node ?? ""),
      relation: String(step.relation ?? step.edge ?? ""),
    }));
  }

  /**
   * Return KB statistics (entity/relation/community counts).
   */
  async stats(): Promise<KBStats> {
    const raw = await this._get("/v1/communities");
    return {
      entities: raw?.node_count ?? 0,
      relations: raw?.edge_count ?? 0,
      relation_types: raw?.relation_types ?? 0,
      communities: Array.isArray(raw?.communities) ? raw.communities.length : 0,
    };
  }

  /**
   * Health check — returns true if the API is reachable and ready.
   */
  async isHealthy(): Promise<boolean> {
    try {
      const raw = await this._get("/v1/health");
      return raw?.status === "ok";
    } catch {
      return false;
    }
  }

  // ── Internal ────────────────────────────────────────────────────────────

  private async _post(path: string, body: unknown): Promise<any> {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), this.timeoutMs);
    try {
      const resp = await fetch(`${this.baseUrl}${path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: ctrl.signal,
      });
      if (!resp.ok) {
        const detail = await resp.json().catch(() => null);
        throw new CerebrumError(
          `CEREBRUM API error ${resp.status}`,
          resp.status,
          detail,
        );
      }
      return resp.json();
    } finally {
      clearTimeout(timer);
    }
  }

  private async _get(path: string): Promise<any> {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), this.timeoutMs);
    try {
      const resp = await fetch(`${this.baseUrl}${path}`, { signal: ctrl.signal });
      if (!resp.ok) {
        throw new CerebrumError(`CEREBRUM API error ${resp.status}`, resp.status);
      }
      return resp.json();
    } finally {
      clearTimeout(timer);
    }
  }

  private _toResult(raw: any, query: string, clientElapsedMs: number): Result {
    const answers: any[] = raw?.answers ?? raw?.results ?? [];
    const best = answers[0];

    const trace_path: TraceStep[] = (raw?.trace ?? raw?.path ?? []).map((s: any) => ({
      entity: String(s.entity ?? s.node ?? ""),
      relation: String(s.relation ?? s.edge ?? ""),
    }));

    const top_k: TopCandidate[] = answers.map((a: any) => ({
      entity: String(a.entity_id ?? a.entity ?? a.id ?? ""),
      confidence: Number(a.score ?? a.confidence ?? 0),
    }));

    return {
      answer: best ? String(best.entity_id ?? best.entity ?? best.id ?? "") : "",
      confidence: best ? Number(best.score ?? best.confidence ?? 0) : 0,
      trace_path,
      top_k,
      elapsed_ms: Number(raw?.elapsed_ms ?? clientElapsedMs),
      query,
    };
  }
}

export default Cerebrum;
