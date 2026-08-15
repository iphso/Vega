/** POST /physics + poll GET /physics/jobs/{id}. Never blocks the UI. */

// Overridable so the viewer is not pinned to a service on this machine:
// set PUBLIC_PHYSICS_URL in viewer/.env (or the environment astro build
// runs in) to point at a physics service elsewhere.
export const PHYSICS_URL =
  import.meta.env.PUBLIC_PHYSICS_URL || 'http://localhost:8090';

export interface PhysicsRequestOpts {
  /** Progress text for the status line: 'loading…', 'cached', 'ready'. */
  onStatus?: (msg: string) => void;
  signal?: AbortSignal;
}

const cache = new Map<string, any>();

function cacheKey(body: unknown): string {
  return JSON.stringify(body);
}

export async function requestPhysics(
  body: unknown,
  opts: PhysicsRequestOpts = {},
): Promise<any> {
  const key = cacheKey(body);
  if (cache.has(key)) return cache.get(key);

  const onStatus = opts.onStatus || (() => {});
  const signal = opts.signal;

  onStatus('loading…');
  const post = await fetch(`${PHYSICS_URL}/physics`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  });

  if (post.status === 200) {
    const bundle = await post.json();
    cache.set(key, bundle);
    onStatus(bundle.meta?.cached ? 'cached' : 'ready');
    return bundle;
  }
  if (post.status !== 202) {
    const err = await post.text();
    throw new Error(`physics POST ${post.status}: ${err}`);
  }

  const { job_id } = await post.json();
  onStatus('loading… polling');
  while (true) {
    if (signal?.aborted) throw new DOMException('aborted', 'AbortError');
    await new Promise((r) => setTimeout(r, 1500));
    const get = await fetch(`${PHYSICS_URL}/physics/jobs/${job_id}`, { signal });
    if (get.status === 202) {
      onStatus('loading…');
      continue;
    }
    if (!get.ok) {
      const err = await get.text();
      throw new Error(`physics job ${get.status}: ${err}`);
    }
    const bundle = await get.json();
    cache.set(key, bundle);
    onStatus(bundle.meta?.cached ? 'cached' : 'ready');
    return bundle;
  }
}

export function peekPhysicsCache(body: unknown): any {
  return cache.get(cacheKey(body));
}
