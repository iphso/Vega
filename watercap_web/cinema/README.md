# cinema: animated, captioned 3D explainers for simulations

`cinema.js` turns a declarative **script** into a deterministic film: a three.js scene rendered through
an HDR pipeline (bloom, ACES tone mapping, vignette, grain), camera moves on time-aware Catmull-Rom splines,
captions, and callouts that follow 3D points or moving particles. Every frame is a pure function of the
timeline time `t`, so `?t=12.5&paused=1` always renders the same frame, which is useful for screenshots and tests.

- `cinema.js` is the engine (ES module, no build step). The page needs an import map that resolves `"three"`.
- `player.html?script=<file>.js` plays any script full-window. Add `&t=…&paused=1` for one frame, `&bare=1` for the stage alone.
- `watercap_script.js` (with `watercap_physics.js` and `watercap_scene.json`) is the watercap film, embedded in `../validation.html`.

```js
import { createCinema } from "./cinema/cinema.js";
import script from "./cinema/my_script.js";
const film = await createCinema(document.getElementById("film"), script);   // film.seek(t), .play(), .pause(), .record()
```

## A script

```js
export default {
  title: "My simulation", fileName: "my-film",           // fileName is used for "Save video"
  async setup(ctx) { return await (await fetch("data.json")).json(); },   // becomes ctx.data
  layers: { … },                                          // what is drawn (see below)
  shots: [ … ] or (ctx) => [ … ],                         // when, from where, with what visible
  captions: [ … ], callouts: [ … ],                       // either may be (ctx) => [ … ]
  status: { pass: { color: "#7ee0a8", label: "Validated" }, … },   // chip colours for callouts and captions
  legend: [{ color, label }, …],                          // shown when the param "legend" is > 0
  corner: t => ["left corner text", "right corner text"],
  overlay(g, t, ui) { … },                                // optional extra 2D drawing (design units 1600 × 900)
  defaults: { myParam: 0 },                               // params before any shot sets them
  post: { bloom: 0.9, threshold: 0.9, exposure: 1, vignette: 0.4, grain: 0.9 },
  fade: 1.2,                                              // default crossfade (s) between shots
  posterTime: 80,                                         // frame shown to prefers-reduced-motion users
};
```

### Shots
Shots play in order. Each one has:

- `duration` (s). Use `id` so callouts and captions can refer to it, and `chapter: "Title"` to start a chapter (scrubber tick and chapter button).
- `camera: [{ at, target: [x,y,z], az, el, dist, fov, ease?, cut? }]`. `at` is in shot-local seconds. `az` and `el` are in degrees around `target`; you can give `pos: [x,y,z]` instead. Keys from all shots form one spline, so moves flow across shot boundaries. `ease: true` stops the camera at that key, and `cut: true` makes a hard cut.
- `layers: { name: weight }`. A layer that a shot doesn't list has weight 0. Weights crossfade over `fade` seconds at shot boundaries.
- `params: { name: value }`. A value is a number, or keyframes `[[at, v], …]` with smoothstep between keys. A param that a shot doesn't set keeps its value from the previous shot. The engine reads `shiftX` (lens shift; positive moves the subject right, as a fraction of the width), `legend`, `hud`, `bloom` and `exposure`. Every other param is yours.
- `titleCard: { kicker, title, sub, meta, in, out }` and `caption: { kicker, title, line, status }` (optional).

### Layers
A layer is either a function `(scene, t, dt, params) => {}` or an object `{ init(ctx), update(scene, t, dt, params) }`.
`params` is `{ w, t, shot, ctx, get(name, default) }`, where `w` is the layer's crossfaded weight. Set `visible = w > 0`
and scale opacities by `w`. **Make each update a pure function of `t`** (use `ctx.hash01(i, k)` or a seeded
`ctx.mulberry32`, never `Math.random`). For history, look up params at other times with `ctx.param(name, tPast)`.
For example, "was the spray on when this drop was emitted?" Precompute heavy physics in `setup`/`init` (see
`watercap_physics.js`) and play the tables back.

Helpers exported by `cinema.js` include `softPoints(n)` (soft additive sprites with per-point colour, alpha and size in
world metres; HDR colours above 1 bloom), `trailSegments(n)` (fading line segments for motion trails), `smooth`, `clamp`, `lerp`,
`hash01` and `mulberry32`.

### Callouts and captions
```js
{ at: [x,y,z] | (t, ctx) => Vector3 | null,      // null hides it (for example, the particle is gone)
  shot: "spray", in: 1.0, out: 6.0,              // or absolute t0 / t1
  label: "Drop heating", detail: "…" | (t, ctx) => "…", status: "fail",
  side: "left" | "right", dy: 0, style: "compact", maxLines: 2 }
```
Callouts are projected every frame. They stack in a column on each side without overlapping, stay on screen,
and draw their leader line and box in and out. On screens narrower than 640 px they collapse into numbered dots,
with a list under the player. Captions (`{ shot, in, out, kicker, title, line, status }`) cross-fade at the lower left.
To follow a particle, give the layer a `track(i)` that returns `(t, ctx) => position`, and pick `i` in
`callouts(ctx)` by searching for a particle that is in view for the whole window (see `spray.find`).

## Transport, testing, export
Space plays or pauses, ← → seek 5 s (Shift for 1 s), ↑ ↓ or [ ] change chapter, L toggles loop. Clicking the stage also toggles play.
Headless frames:
```
google-chrome --headless=new --no-sandbox --use-gl=angle --use-angle=swiftshader --enable-unsafe-swiftshader \
  --window-size=1600,987 --virtual-time-budget=20000 --screenshot=f.png 'http://localhost:8000/cinema/player.html?bare=1&t=20&paused=1'
```
Serve over HTTP, because ES modules don't load from `file://`. "Save video" re-renders the WebGL canvas and the overlay at 1920×1080,
plays from 0 in real time and downloads a WebM through MediaRecorder (VP9 where the browser supports it).
