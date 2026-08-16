/** Small viridis/plasma LUTs. Sampled stops, linearly interpolated. */

const VIRIDIS = [
  [68, 1, 84],
  [72, 40, 120],
  [62, 74, 137],
  [49, 104, 142],
  [38, 130, 142],
  [31, 158, 137],
  [53, 183, 121],
  [109, 205, 89],
  [180, 222, 44],
  [253, 231, 37],
];

const PLASMA = [
  [13, 8, 135],
  [84, 2, 163],
  [139, 10, 165],
  [185, 50, 137],
  [219, 92, 104],
  [244, 136, 73],
  [254, 188, 43],
  [240, 249, 33],
];

export const COLORMAP_NAMES = ['viridis', 'plasma'];

function lerpStops(stops, frac) {
  const f = Math.max(0, Math.min(1, frac));
  const scaled = f * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(scaled));
  const t = scaled - i;
  const a = stops[i], b = stops[i + 1];
  return [
    Math.round(a[0] + (b[0] - a[0]) * t),
    Math.round(a[1] + (b[1] - a[1]) * t),
    Math.round(a[2] + (b[2] - a[2]) * t),
  ];
}

export function colormapRgb(name, frac) {
  const stops = name === 'plasma' ? PLASMA : VIRIDIS;
  return lerpStops(stops, frac);
}

export function colormapHex(name, frac) {
  const [r, g, b] = colormapRgb(name, frac);
  return `#${((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1)}`;
}
