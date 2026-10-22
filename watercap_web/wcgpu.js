// WebGPU port of wcmodel.js: one invocation = one design, f32 throughout.
// The shader mirrors wcmodel.js function for function; the page checks the
// two against each other on random designs before trusting the GPU.
(function (root) {
  "use strict";
  const IN = ["air_flow_m3_s", "tube_diameter_m", "spray_section_length_m", "corona_section_length_m", "nozzle_height_frac",
    "disc_diameter_frac", "disc_spacing_m", "spray_flow_L_min", "spray_T_C", "spray_d_um", "spray_q_frac", "corona_voltage_kV",
    "corona_polarity", "spray_polarity_rel", "inlet_T_C", "inlet_RH", "inlet_lwc_g_m3", "mist_mmd_um", "mist_gsd", "spray_gsd", "spray_v0"];
  const OUT = ["outlet_total_water_frac", "vapor_removed_frac", "liquid_escape_frac", "recovered_L_h", "electrical_power_W",
    "fan_power_W", "arc_margin", "corona_current_mA", "vapor_ceiling_frac", "pressure_drop_Pa", "wall_field_kV_cm", "corona_onset_kV"];
  const IN_STRIDE = 24, OUT_STRIDE = 12;

  function packFit(fit) {
    const t = fit.tables, parts = [], off = {};
    const push = (name, arr) => { off[name] = parts.length; for (const v of arr) parts.push(v); };
    push("a0", t.lnVon.axes[0]); push("a1", t.lnVon.axes[1]); push("a2", t.lnVon.axes[2]);
    push("von", t.lnVon.vals); push("ew", t.lnEwall.vals);
    push("j0", t.lnJ.axes[0]); push("j1", t.lnJ.axes[1]); push("j2", t.lnJ.axes[2]); push("jv", t.lnJ.vals);
    push("sc", [fit.cE, fit.cN, fit.kq || 1, fit.quench || 0, fit.mu_ref]);
    const n = [t.lnVon.axes[0].length, t.lnVon.axes[1].length, t.lnVon.axes[2].length, t.lnJ.axes[0].length, t.lnJ.axes[1].length, t.lnJ.axes[2].length];
    return { data: new Float32Array(parts), off, n, jex: t.lnJ.extrap };
  }

  function shaderSource(pf) {
    const o = pf.off, n = pf.n;
    return /* wgsl */`
struct Params { count: u32, nb: i32, nz: i32, outer: i32, mode: i32, nsubD: i32, dtf: f32, pad: f32 };
@group(0) @binding(0) var<storage, read> designs: array<f32>;
@group(0) @binding(1) var<storage, read_write> outs: array<f32>;
@group(0) @binding(2) var<storage, read> fit: array<f32>;
@group(0) @binding(3) var<uniform> P: Params;

const PI: f32 = 3.14159265;
const P_ATM: f32 = 101325.0; const R_D: f32 = 287.05; const R_V: f32 = 461.5; const R_U: f32 = 8.314; const M_W: f32 = 0.018015;
const CP_A: f32 = 1006.0; const CP_V: f32 = 1860.0; const CP_L: f32 = 4186.0; const RHO_L: f32 = 998.0; const EPS0: f32 = 8.854e-12;
const EPS_R_W: f32 = 80.0; const E_CHG: f32 = 1.602e-19; const K_B: f32 = 1.381e-23; const E_OVER_KB: f32 = 11600.29; const G: f32 = 9.81;
const MU_G: f32 = 1.81e-5; const K_G: f32 = 0.026; const PR: f32 = 0.71; const SC: f32 = 0.60;
const DISC_EDGE: f32 = 1.5e-4; const SPRAY_V0: f32 = 2.0; const FAN_EFF: f32 = 0.6; const E_BD: f32 = 2.4e6;
const NZM: i32 = 64; const NBM: i32 = 6;

fn cbrt(x: f32) -> f32 { return pow(max(x, 1e-37), 1.0 / 3.0); }
fn psat(T0: f32) -> f32 { let T = clamp(T0, -60.0, 150.0); return 611.21 * exp((18.678 - T / 234.5) * (T / (257.14 + T))); }
fn dpsat(T0: f32) -> f32 { let T = clamp(T0, -60.0, 150.0); let u = T / (257.14 + T);
  return psat(T) * (-u / 234.5 + (18.678 - T / 234.5) * 257.14 / ((257.14 + T) * (257.14 + T))); }
fn w_from_pv(pv: f32) -> f32 { return 0.622 * pv / (P_ATM - pv); }
fn pv_from_w(w: f32) -> f32 { return w * P_ATM / (0.622 + w); }
fn h_fg(T: f32) -> f32 { return 2.501e6 - 2370.0 * T; }
fn sigma(T: f32) -> f32 { return 0.0756 - 1.4e-4 * T; }
fn Dv(T: f32) -> f32 { return 2.11e-5 * pow((T + 273.15) / 273.15, 1.94); }
fn rho_gas(T: f32, w: f32) -> f32 { let pv = pv_from_w(w); let Tk = T + 273.15; return (P_ATM - pv) / (R_D * Tk) + pv / (R_V * Tk); }
fn kt_factor(a: f32, q: f32, T: f32) -> f32 {
  let qa = q / (a * a);
  let term = 2.0 * sigma(T) / a - (1.0 - 1.0 / EPS_R_W) * qa * qa / (32.0 * PI * PI * EPS0);
  return exp(clamp(M_W / (RHO_L * R_U * (T + 273.15)) * term, -5.0, 5.0));
}
fn rayleigh(a: f32, T: f32) -> f32 { return 8.0 * PI * sqrt(EPS0 * sigma(T) * a) * a; }
fn v_term(a: f32, rho: f32) -> f32 {
  var v = 2.0 * RHO_L * G * a * a / (9.0 * MU_G);
  for (var i = 0; i < 8; i++) { let re = rho * v * 2.0 * a / MU_G; v = 2.0 * RHO_L * G * a * a / (9.0 * MU_G * (1.0 + 0.15 * pow(max(re, 1e-12), 0.687))); }
  return v;
}
fn cunn(a: f32) -> f32 { let kn = 0.066e-6 / a; return 1.0 + kn * (1.257 + 0.4 * exp(-1.1 / kn)); }
fn nu_sh(a: f32, vrel: f32, rho: f32) -> vec2<f32> { let s = 0.6 * sqrt(rho * abs(vrel) * 2.0 * a / MU_G); return vec2<f32>(2.0 + s * pow(PR, 1.0 / 3.0), 2.0 + s * pow(SC, 1.0 / 3.0)); }
fn drop_Teq(a: f32, q: f32, Tg: f32, pv: f32, vrel: f32, rho: f32) -> f32 {
  let ns = nu_sh(a, vrel, rho); let D = Dv(Tg); let rvi = pv / (R_V * (Tg + 273.15));
  var Td = Tg - 0.5;
  for (var i = 0; i < 6; i++) {
    let Tk = Td + 273.15; let kt = kt_factor(a, q, Td); let rvs = psat(Td) * kt / (R_V * Tk);
    let drho = dpsat(Td) * kt / (R_V * Tk) - rvs / Tk;
    let f = ns.x * K_G * (Tg - Td) + h_fg(Td) * ns.y * D * (rvi - rvs); let df = -ns.x * K_G - h_fg(Td) * ns.y * D * drho;
    Td = Td - f / df;
  }
  return Td;
}
fn drop_rates(a: f32, q: f32, Td: f32, Tg: f32, pv: f32, vrel: f32, rho: f32) -> vec2<f32> {
  let ns = nu_sh(a, vrel, rho); let D = Dv(Tg);
  let rvi = pv / (R_V * (Tg + 273.15)); let rvs = psat(Td) * kt_factor(a, q, Td) / (R_V * (Td + 273.15));
  return vec2<f32>(2.0 * PI * a * ns.y * D * (rvi - rvs), 2.0 * PI * a * ns.x * K_G * (Tg - Td));
}
fn evap_cut(md: f32, m: f32, mref: f32) -> f32 { if (md >= 0.0) { return md; } let r = m / (m + 1e-6 * mref); return md * r * r; }
fn spark_V(gap: f32, pol: f32) -> f32 { let gc = gap * 100.0; let e = select(4.5, 8.0, pol < 0.0); return 1e3 * e * (1.0 + 1.0 / sqrt(gc)) * gc; }
fn ppf(p: f32) -> f32 {
  if (p < 0.02425 || p > 0.97575) {
    let pp = min(p, 1.0 - p); let q = sqrt(-2.0 * log(pp));
    let r = (((((-7.784894002430293e-3 * q - 3.223964580411365e-1) * q - 2.400758277161838) * q - 2.549732539343734) * q + 4.374664141464968) * q + 2.938163982698783) /
            ((((7.784695709041462e-3 * q + 3.224671290700398e-1) * q + 2.445134137142996) * q + 3.754408661907416) * q + 1.0);
    return select(-r, r, p < 0.5);
  }
  let q = p - 0.5; let r = q * q;
  return (((((-3.969683028665376e1 * r + 2.209460984245205e2) * r - 2.759285104469687e2) * r + 1.383577518672690e2) * r - 3.066479806614716e1) * r + 2.506628277459239) * q /
         (((((-5.447609879822406e1 * r + 1.615858368580409e2) * r - 1.556989798598866e2) * r + 6.680131188771972e1) * r - 1.328068155288572e1) * r + 1.0);
}
// trilinear lattice lookup with per-table axis offsets
fn axis_cell(off: u32, n: u32, v: f32, ex: f32) -> vec2<f32> {
  var i: u32 = 0u;
  loop { if (i >= n - 2u) { break; } if (v > fit[off + i + 1u]) { i = i + 1u; } else { break; } }
  let g0 = fit[off + i]; let g1 = fit[off + i + 1u];
  return vec2<f32>(f32(i), clamp((v - g0) / (g1 - g0), -ex, 1.0 + ex));
}
fn lattice(o0: u32, n0: u32, o1: u32, n1: u32, o2: u32, n2: u32, ov: u32, v0: f32, v1: f32, v2: f32, ex: f32) -> f32 {
  let c0 = axis_cell(o0, n0, v0, ex); let c1 = axis_cell(o1, n1, v1, ex); let c2 = axis_cell(o2, n2, v2, ex);
  let i0 = u32(c0.x); let i1 = u32(c1.x); let i2 = u32(c2.x);
  var out = 0.0;
  for (var a = 0u; a < 2u; a++) { for (var b = 0u; b < 2u; b++) { for (var c = 0u; c < 2u; c++) {
    let wa = select(1.0 - c0.y, c0.y, a == 1u); let wb = select(1.0 - c1.y, c1.y, b == 1u); let wc = select(1.0 - c2.y, c2.y, c == 1u);
    out += wa * wb * wc * fit[ov + ((i0 + a) * n1 + i1 + b) * n2 + i2 + c];
  } } }
  return out;
}
struct CS { V_on: f32, I: f32, E: f32, Ech: f32, N: f32, mu: f32 };
fn corona_eval(V_on0: f32, Ewpv: f32, J: f32, R: f32, s: f32, mu: f32, V: f32, dVon: f32, refit: bool) -> CS {
  var c: CS;
  c.mu = mu; c.V_on = V_on0 + dVon;
  if (V <= c.V_on) { c.I = 0.0; c.E = V * Ewpv; c.Ech = c.E; c.N = 0.0; return c; }
  let xr = V / c.V_on; c.I = J * xr * (xr - 1.0) / s;
  let El = V * Ewpv;
  c.E = fit[${o.sc}u] * sqrt(El * El + c.I / (2.0 * PI * EPS0 * mu));
  c.Ech = fit[${o.sc + 2}u] * c.E;
  c.N = fit[${o.sc + 1}u] * c.I / (2.0 * PI * R) / (E_CHG * mu * c.E);
  return c;
}
fn corona_rom(V: f32, R: f32, rd: f32, s: f32, mu: f32) -> CS {
  var c: CS; c.mu = mu;
  let gap = R - rd; let shield = 1.0 + exp(-2.0 * s / max(gap, 1e-6));
  let E_on = 3.1e6 * (1.0 + 0.0308 / sqrt(DISC_EDGE));
  c.V_on = E_on * DISC_EDGE * log(2.0 * gap / DISC_EDGE) * shield;
  let lnr = log(R / max(rd, 1e-4));
  if (V <= c.V_on) { c.I = 0.0; c.E = V / (R * lnr); c.Ech = c.E; c.N = 0.0; return c; }
  c.I = 8.0 * PI * EPS0 * mu * V * (V - c.V_on) / (R * R * lnr);
  let El = c.V_on / (R * lnr); c.E = sqrt(El * El + c.I / (2.0 * PI * EPS0 * mu)); c.Ech = c.E;
  c.N = c.I / (2.0 * PI * R) / (E_CHG * mu * c.E);
  return c;
}
// exact charge step: returns (q_new, avg q_rel while >= 0, avg |q_rel| while < 0)
fn charge_step(a: f32, q: f32, cs: CS, pol: f32, T: f32, dt: f32) -> vec3<f32> {
  let qr0 = pol * q;
  if (cs.N <= 0.0 || dt <= 0.0) { return vec3<f32>(q, max(qr0, 0.0), max(-qr0, 0.0)); }
  let p = 3.0 * EPS_R_W / (EPS_R_W + 2.0); let qs = 4.0 * PI * EPS0 * p * cs.Ech * a * a;
  let tau = 4.0 * EPS0 / (cs.N * E_CHG * cs.mu);
  let Tk = T + 273.15; let cbar = sqrt(8.0 * K_B * Tk / (PI * 4.8e-26)); let cy = E_OVER_KB / (4.0 * PI * EPS0 * a * Tk);
  let D0 = PI * a * a * cs.N * E_CHG * cbar;
  var qr = qr0; var t = dt; var accP = 0.0; var accN = 0.0;
  if (qr < 0.0) {
    let lam = 4.0 / tau + D0 * cy; let qst = (qs / tau + D0) / lam;
    let t0 = log((qst - qr) / qst) / lam; let tt = min(t, t0); let e = exp(-lam * tt);
    accN -= qst * tt + (qr - qst) * (1.0 - e) / lam;
    qr = select(0.0, qst + (qr - qst) * e, tt < t0); t -= tt;
  }
  if (t > 0.0) {
    if (qr < qs) {
      let t1 = tau * qr / (qs - qr); let u = t / (t1 + tau);
      if (u < 1e-3) { accP += qs * (t * t1 / (t1 + tau) + tau * u * u * (0.5 - u / 3.0)); } else { accP += qs * (t - tau * log(1.0 + u)); }
      qr = qs * (t1 + t) / (t1 + t + tau);
    } else { accP += qr * t; }
  }
  var dqd = 0.0;
  if (t > 0.0) { dqd = min(D0 * exp(-min(qr * cy, 80.0)) * t, 1.0 / cy); }
  qr += dqd;
  return vec3<f32>(pol * qr, max(accP / dt + select(0.0, 0.5 * dqd, qr >= 0.0), 0.0), max(accN / dt, 0.0));
}
fn eE1(c: f32) -> f32 {
  if (c <= 1.0) { return exp(c) * (-log(c) - 0.57721566 + c * (0.99999193 + c * (-0.24991055 + c * (0.05519968 + c * (-0.00976004 + c * 0.00107857))))); }
  return (c * c + 2.334733 * c + 0.250621) / (c * (c * c + 3.330657 * c + 1.681534));
}
fn exch_weight(ke: f32, kl: f32, dt: f32, hyper: bool) -> f32 {
  if (kl * dt < 1e-7) { return 1.0; }
  var den = dt; if (ke * dt > 1e-6) { den = (1.0 - exp(-ke * dt)) / ke; }
  if (!hyper) { var nm = dt; if ((ke + kl) * dt > 1e-6) { nm = (1.0 - exp(-(ke + kl) * dt)) / (ke + kl); } return nm / den; }
  if (ke * dt < 1e-5) { return log(1.0 + kl * dt) / (kl * dt); }
  let c0 = ke / kl; let c1 = c0 * (1.0 + kl * dt);
  return clamp((eE1(c0) - exp(-ke * dt) * eE1(c1)) / kl / den, 0.0, 1.0);
}
fn max_vapor_removal(mair: f32, Tin: f32, win: f32, ms: f32, Ts: f32) -> f32 {
  if (ms <= 0.0 || Ts >= Tin) { return 0.0; }
  var lo = Ts; var hi = Tin;
  for (var i = 0; i < 40; i++) {
    let mid = 0.5 * (lo + hi); let wf = min(w_from_pv(psat(mid)), win);
    let rel = mair * (CP_A + win * CP_V) * (Tin - mid) + h_fg(mid) * mair * (win - wf);
    if (rel - ms * CP_L * (mid - Ts) > 0.0) { lo = mid; } else { hi = mid; }
  }
  return max(mair * (win - w_from_pv(psat(0.5 * (lo + hi)))), 0.0);
}
fn mass_r(m: f32) -> f32 { return cbrt(max(m, 1e-30) / (RHO_L * 4.0 / 3.0 * PI)); }

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let id = gid.x;
  if (id >= P.count) { return; }
  let b0 = id * ${IN_STRIDE}u;
  var X: array<f32, ${IN_STRIDE}>;
  for (var i = 0u; i < ${IN_STRIDE}u; i++) { X[i] = designs[b0 + i]; }
  let nb = P.nb; let nz = P.nz;
  let pol_c = select(-1.0, 1.0, X[12] >= 0.0); let pol_s = select(-pol_c, pol_c, X[13] >= 0.0);
  let R = X[1] / 2.0; let A = PI * R * R; let Ls = X[2]; let Lc = X[3]; let L = Ls + Lc;
  let z_noz = X[4] * Ls; let rd = X[5] * R; let sp = X[6];
  let T_in = X[14]; let w_in = w_from_pv(X[15] * psat(T_in)); let rho_in = rho_gas(T_in, w_in);
  let mair = rho_in * X[0] / (1.0 + w_in); let U = X[0] / A;
  let V_c = X[11] * 1e3; let mu = select(1.4e-4, 2.1e-4, pol_c < 0.0);
  let refit = P.mode == 0;
  var V_on0 = 0.0; var Ewpv = 0.0; var Jb = 0.0; var cs: CS;
  if (refit) {
    let lR = log(R); let lS = log(sp); let fr = rd / R;
    V_on0 = exp(lattice(${o.a0}u, ${n[0]}u, ${o.a1}u, ${n[1]}u, ${o.a2}u, ${n[2]}u, ${o.von}u, lR, fr, lS, 0.0));
    Ewpv = exp(lattice(${o.a0}u, ${n[0]}u, ${o.a1}u, ${n[1]}u, ${o.a2}u, ${n[2]}u, ${o.ew}u, lR, fr, lS, 0.0));
    Jb = exp(lattice(${o.j0}u, ${n[3]}u, ${o.j1}u, ${n[4]}u, ${o.j2}u, ${n[5]}u, ${o.jv}u, lR, log(fr), lS, ${pf.jex.toFixed(3)})) * (mu / fit[${o.sc + 4}u]);
    cs = corona_eval(V_on0, Ewpv, Jb, R, sp, mu, V_c, 0.0, true);
  } else { cs = corona_rom(V_c, R, rd, sp, mu); }
  let quench = select(0.0, fit[${o.sc + 3}u], refit);
  let I_corona = cs.I * Lc;
  let arc_c = 1.0 - V_c / spark_V(R - rd, pol_c);

  // spray classes
  let Qtot = X[7] / 60000.0; let a_s = X[9] * 0.5e-6; let q_s = pol_s * X[10] * rayleigh(a_s, X[8]);
  // spray classes: lognormal (4 equal-volume bins, charge a fixed fraction of
  // each bin's Rayleigh limit) when a spread is given, else the ROM's
  // electrospray split (85% main drops, 15% satellites at a/2, q/4)
  var sp_a: array<f32, 4>; var sp_q: array<f32, 4>; var sp_f: array<f32, 4>;
  if (X[19] > 1.0001) {
    for (var c = 0; c < 4; c++) {
      sp_a[c] = X[9] * 0.5e-6 * exp(log(X[19]) * ppf((f32(c) + 0.5) / 4.0));
      sp_q[c] = pol_s * X[10] * rayleigh(sp_a[c], X[8]); sp_f[c] = 0.25;
    }
  } else {
    sp_a = array<f32, 4>(a_s, a_s / 2.0, a_s, a_s); sp_q = array<f32, 4>(q_s, q_s / 4.0, 0.0, 0.0); sp_f = array<f32, 4>(0.85, 0.15, 0.0, 0.0);
  }
  let v0s = X[20];
  let spray_in = RHO_L * Qtot;
  var isEnt = array<bool, 4>(false, false, false, false); var isFall = array<bool, 4>(false, false, false, false);
  var em0: array<f32, 4>; var Nde: array<f32, 4>; var vkf: array<f32, 4>;
  for (var c = 0; c < 4; c++) {
    let Qc = sp_f[c] * Qtot;
    em0[c] = RHO_L * 4.0 / 3.0 * PI * sp_a[c] * sp_a[c] * sp_a[c]; Nde[c] = RHO_L * Qc / em0[c];
    if (Qc > 0.0) { if (v_term(sp_a[c], rho_in) < 0.8 * U) { isEnt[c] = true; } else { isFall[c] = true; } }
  }
  let anyFall = isFall[0] || isFall[1] || isFall[2] || isFall[3]; let anyEnt = isEnt[0] || isEnt[1] || isEnt[2] || isEnt[3];
  // mist bins
  var m0: array<f32, NBM>; var Ndm: array<f32, NBM>;
  for (var b = 0; b < nb; b++) {
    let d = X[17] * 1e-6 * exp(log(max(X[18], 1.000001)) * ppf((f32(b) + 0.5) / f32(nb)));
    m0[b] = RHO_L * PI * d * d * d / 6.0; Ndm[b] = (X[16] * 1e-3 / f32(nb)) / m0[b] * X[0];
  }
  let dz = L / f32(nz);
  var Tprof: array<f32, NZM>; var wprof: array<f32, NZM>; var S_v: array<f32, NZM>; var S_h: array<f32, NZM>;
  var rqf: array<f32, NZM>; var rqe: array<f32, NZM>; var fd: array<array<f32, NZM>, 4>;
  for (var k = 0; k < nz; k++) { Tprof[k] = T_in; wprof[k] = w_in; S_v[k] = 0.0; S_h[k] = 0.0; rqf[k] = 0.0; rqe[k] = 0.0; fd[0][k] = 0.0; fd[1][k] = 0.0; fd[2][k] = 0.0; fd[3][k] = 0.0; }
  var f_top = 0.0; var f_bottom = 0.0; var f_wall = 0.0; var f_elec = 0.0; var f_drag = 0.0;
  // march outputs
  var o_w = w_in; var o_top = 0.0; var o_wall = 0.0; var o_elec = 0.0; var o_mistOut = 0.0; var o_mistIn = 0.0; var o_mistDep = 0.0; var o_maxRq = 0.0; var o_I = 0.0;
  let nIter = select(1, P.outer, anyFall);
  for (var it = 0; it < nIter; it++) {
    if (anyFall) {
      // ---- falling-spray trace against the current profile (PSI-cell)
      var Sv: array<f32, NZM>; var Sh: array<f32, NZM>; var rq: array<f32, NZM>;
      for (var k = 0; k < nz; k++) { Sv[k] = 0.0; Sh[k] = 0.0; rq[k] = 0.0; fd[0][k] = 0.0; fd[1][k] = 0.0; fd[2][k] = 0.0; fd[3][k] = 0.0; }
      var s_top = 0.0; var s_bottom = 0.0; var s_wall = 0.0; var s_elec = 0.0; var s_drag = 0.0;
      for (var c = 0; c < 4; c++) {
        if (!isFall[c]) { continue; }
        let Nd = Nde[c];
        vkf[c] = U - v_term(sp_a[c], rho_gas(Tprof[0], wprof[0]));
        var z = z_noz; var v = -v0s; var a = sp_a[c]; var q = sp_q[c]; var f = 1.0;
        var Td = Tprof[min(i32(floor(z / dz)), nz - 1)];
        for (var steps = 0; steps < 4000; steps++) {
          if (!(z >= 0.0 && z < L && f > 1e-6)) { break; }
          let k = min(i32(floor(z / dz)), nz - 1); let Tg = Tprof[k]; let wg = wprof[k];
          let rho = rho_gas(Tg, wg); let pv = pv_from_w(wg); let vrel = U - v; let re = rho * abs(vrel) * 2.0 * a / MU_G;
          let taup = RHO_L * 4.0 * a * a / (18.0 * MU_G * (1.0 + 0.15 * pow(max(re, 1e-12), 0.687))); let m = RHO_L * 4.0 / 3.0 * PI * a * a * a;
          let Teq = drop_Teq(a, q, Tg, pv, vrel, rho); let ns = nu_sh(a, vrel, rho);
          let dpdT = dpsat(Td) / (R_V * (Td + 273.15));
          let tauT = max(m * CP_L / (2.0 * PI * a * (ns.x * K_G + h_fg(Td) * ns.y * Dv(Tg) * dpdT)), 1e-9);
          let Zc = cunn(a) / (6.0 * PI * MU_G * a);
          var dt = P.dtf * dz / max(abs(v), 0.05 * U); var dq = 0.0; var toE = 0.0; var kl = 0.0;
          if (z >= Ls) {
            let r3 = charge_step(a, q, cs, pol_c, Tg, dt); dq = (r3.x - q) / dt;
            kl = 2.0 * (r3.y + r3.z) * cs.E * Zc / R; toE = r3.z / max(r3.y + r3.z, 1e-30);
          } else { kl = abs(q) * Zc * abs(rqf[k] + rqe[k]) / EPS0; }
          var eTx = exp(-dt / tauT); var Tavg = Teq + (Td - Teq) * tauT / dt * (1.0 - eTx);
          var rr = drop_rates(a, q, Tavg, Tg, pv, vrel, rho);
          if (abs(rr.x) * dt > 0.2 * m) { dt = 0.2 * m / abs(rr.x); eTx = exp(-dt / tauT); Tavg = Teq + (Td - Teq) * tauT / dt * (1.0 - eTx); rr = drop_rates(a, q, Tavg, Tg, pv, vrel, rho); }
          let Tnew = Teq + (Td - Teq) * eTx; let ek = exp(-kl * dt); let fn_ = f * ek;
          var favg = f; if (kl * dt > 1e-7) { favg = f * (1.0 - ek) / (kl * dt); }
          let veq = U - G * taup; let ep = exp(-dt / taup); let vn = veq + (v - veq) * ep; let dzp = veq * dt + (v - veq) * taup * (1.0 - ep);
          let mn = max(m + rr.x * dt, 1e-30); let cond = rr.x * Nd * favg * dt; let lost = Nd * (f * m - fn_ * mn) + cond;
          s_elec += toE * lost; s_wall += (1.0 - toE) * lost;
          Sv[k] -= cond; Sh[k] -= rr.y * Nd * favg * dt; s_drag += Nd * favg * m * (vn - v + G * dt);
          fd[c][k] += Nd * favg * dt;
          rq[k] += q * Nd * favg * dt;
          a = mass_r(mn); q += dq * dt;
          let qR = rayleigh(a, Tavg); if (abs(q) > qR) { q = sign(q) * qR; }
          Td = Tnew; v = vn; f = fn_; z += dzp;
        }
        let me = RHO_L * 4.0 / 3.0 * PI * a * a * a;
        if (z >= L) { s_top += Nd * f * me; } else if (z < 0.0) { s_bottom += Nd * f * me; } else { s_wall += Nd * f * me; }
      }
      let relax = select(1.0, 0.5, it > 0);
      for (var k = 0; k < nz; k++) {
        for (var c = 0; c < 4; c++) { fd[c][k] /= A * dz; }
        S_v[k] = relax * Sv[k] + (1.0 - relax) * S_v[k]; S_h[k] = relax * Sh[k] + (1.0 - relax) * S_h[k];
        rqf[k] = relax * rq[k] / (A * dz) + (1.0 - relax) * rqf[k];
      }
      f_top = s_top; f_bottom = s_bottom; f_wall = s_wall; f_elec = s_elec; f_drag = s_drag;
    }
    // ---- implicit-in-spirit z-march of gas + mist + entrained spray
    var Tg = T_in; var w = w_in;
    var mm: array<f32, NBM>; var mq: array<f32, NBM>; var mF: array<f32, NBM>; var mD: array<f32, NBM>;
    for (var b = 0; b < nb; b++) { mm[b] = m0[b]; mq[b] = 0.0; mF[b] = 1.0; mD[b] = 0.0; }
    var em = em0; var eT = array<f32, 4>(X[8], X[8], X[8], X[8]); var eq = sp_q;
    var eF = array<f32, 4>(1.0, 1.0, 1.0, 1.0); var eW = array<f32, 4>(); var eE = array<f32, 4>();
    var col: array<vec4<f32>, 8>;
    var maxRq = 0.0; var Iacc = 0.0;
    for (var k = 0; k < nz; k++) {
      let z0 = f32(k) * dz; let z1 = f32(k + 1) * dz;
      var cuts = array<f32, 4>(z0, z1, z1, z1); var ncut = 1;
      var p1 = z1; var p2 = z1;
      if (z_noz > z0 && z_noz < z1) { p1 = z_noz; }
      if (Ls > z0 && Ls < z1) { p2 = Ls; }
      let lo_ = min(p1, p2); let hi_ = max(p1, p2);
      if (lo_ < z1) { cuts[ncut] = lo_; ncut++; }
      if (hi_ < z1 && hi_ > lo_) { cuts[ncut] = hi_; ncut++; }
      cuts[ncut] = z1;
      for (var sgi = 0; sgi < ncut; sgi++) {
        let za = cuts[sgi]; let zb = cuts[sgi + 1]; let zm = 0.5 * (za + zb); let hseg = zb - za;
        if (hseg <= 1e-7 * L) { continue; }
        let corona = zm >= Ls; let sprayOn = anyEnt && zm >= z_noz;
        var csz = cs;
        if (corona && quench > 0.0) {
          let rho = rho_gas(Tg, w); var rp = 0.0;
          for (var b = 0; b < nb; b++) { let a = mass_r(mm[b]); let v = max(U - v_term(a, rho), 0.05 * U); rp += Ndm[b] * mF[b] / (A * v) * mq[b]; }
          if (sprayOn) { for (var c = 0; c < 4; c++) { if (isEnt[c]) { let a = mass_r(em[c]); let v = max(U - v_term(a, rho), 0.05 * U); rp += Nde[c] * eF[c] / (A * v) * eq[c]; } } }
          csz = corona_eval(V_on0, Ewpv, Jb, R, sp, mu, V_c, quench * max(0.0, pol_c * rp) * R * R / (4.0 * EPS0), true);
        }
        if (corona) { Iacc += csz.I * hseg; }
        // sub-steps from the exchange stiffness and the Coulomb dispersion rate
        var lam = 0.0; var lamD = 0.0;
        {
          let rho = rho_gas(Tg, w); let Cg = mair * (CP_A + w * CP_V);
          let dvw = 0.622 * P_ATM / ((0.622 + w) * (0.622 + w)) / (R_V * (Tg + 273.15));
          for (var b = 0; b < nb; b++) {
            let a = mass_r(mm[b]); let vt = v_term(a, rho); let v = max(U - vt, 0.05 * U); let ns = nu_sh(a, vt, rho);
            lam += Ndm[b] * mF[b] / v * 2.0 * PI * a * (ns.x * K_G / Cg + ns.y * Dv(Tg) * dvw / mair);
          }
          if (sprayOn) { for (var c = 0; c < 4; c++) { if (isEnt[c]) {
            let a = mass_r(em[c]); let vt = v_term(a, rho); let v = max(U - vt, 0.05 * U); let ns = nu_sh(a, vt, rho);
            lam += Nde[c] * eF[c] / v * 2.0 * PI * a * (ns.x * K_G / Cg + ns.y * Dv(Tg) * dvw / mair);
            if (!corona) { lamD += abs(eq[c]) * cunn(a) / (6.0 * PI * MU_G * a) * abs(rqf[k] + Nde[c] * eF[c] / (A * v) * eq[c]) / EPS0 / v; }
          } } }
        }
        let nsub = max(clamp(i32(ceil(lam / 0.7 * hseg)), 1, 32), clamp(i32(ceil(lamD / 0.5 * hseg)), 1, P.nsubD));
        let h = hseg / f32(nsub);
        for (var ss = 0; ss < nsub; ss++) {
          let rho = rho_gas(Tg, w); let pv = pv_from_w(w);
          var rqv = rqf[k]; var dW = 0.0; var dH = 0.0; var ncol = 0;
          if (!corona) {
            for (var c = 0; c < 4; c++) { if (isFall[c] && fd[c][k] > 0.0) { col[ncol] = vec4<f32>(fd[c][k], sp_a[c], sp_q[c], vkf[c]); ncol++; } }
          }
          var e_cond = array<f32, 4>(); var e_dm = array<f32, 4>(); var e_dT = array<f32, 4>();
          var e_dq = array<f32, 4>(); var e_dF = array<f32, 4>(); var e_toE = array<f32, 4>();
          var Cl = 0.0; var ClT = 0.0;
          if (sprayOn) {
            var rqE = 0.0;
            for (var c = 0; c < 4; c++) { if (isEnt[c] && eF[c] > 1e-9) {
              let a = mass_r(em[c]); let qR = rayleigh(a, eT[c]); let q = select(eq[c], sign(eq[c]) * qR, abs(eq[c]) > qR);
              let v = max(U - v_term(a, rho), 0.05 * U); rqE += Nde[c] * eF[c] / (A * v) * q;
            } }
            rqv += rqE; if (!corona) { maxRq = max(maxRq, abs(rqv)); }
            for (var c = 0; c < 4; c++) { if (isEnt[c] && eF[c] > 1e-9) {
              let m = max(em[c], 1e-30); let a = mass_r(m); let qR = rayleigh(a, eT[c]); let q = select(eq[c], sign(eq[c]) * qR, abs(eq[c]) > qR);
              let vt = v_term(a, rho); let v = max(U - vt, 0.05 * U); let dt = h / v;
              let T0 = eT[c]; let Teq = drop_Teq(a, q, Tg, pv, vt, rho);
              let r0 = drop_rates(a, q, T0, Tg, pv, vt, rho); let re_ = drop_rates(a, q, Teq, Tg, pv, vt, rho);
              let H0 = r0.y + h_fg(T0) * r0.x;
              var tauT = 1e-9;
              if (abs(T0 - Teq) > 1e-4) { tauT = m * CP_L * (T0 - Teq) / max(-H0 * sign(T0 - Teq), 1e-30) * sign(T0 - Teq); }
              else { let ns = nu_sh(a, vt, rho); tauT = m * CP_L / (2.0 * PI * a * (ns.x * K_G + h_fg(T0) * ns.y * Dv(Tg) * dpsat(T0) / (R_V * (T0 + 273.15)))); }
              if (!(tauT > 0.0)) { tauT = 1e-9; }
              let ex = exp(-dt / tauT); let u = select(1.0, tauT / dt * (1.0 - ex), abs(T0 - Teq) > 1e-4);
              var md = re_.x + (r0.x - re_.x) * u; var qc = re_.y + (r0.y - re_.y) * u;
              let mcut = evap_cut(md, m, em0[c]); if (md != 0.0) { qc *= mcut / md; } md = mcut;
              if (md < 0.0 && -md * dt > 0.5 * m) { qc *= (0.5 * m / dt) / -md; md = -0.5 * m / dt; }
              let Zc = cunn(a) / (6.0 * PI * MU_G * a);
              var kl = 0.0; var dq = 0.0; var ek = 1.0;
              if (corona) {
                let r3 = charge_step(a, q, csz, pol_c, Tg, dt); dq = (r3.x - q) / dt;
                kl = 2.0 * (r3.y + r3.z) * csz.E * Zc / R; e_toE[c] = r3.z / max(r3.y + r3.z, 1e-30); ek = exp(-kl * dt);
              } else {
                kl = abs(q) * Zc * abs(rqv) / EPS0;
                if (ncol < 4) { col[ncol] = vec4<f32>(Nde[c] * eF[c] / (A * v), a, q, v); ncol++; }
                ek = 1.0 / (1.0 + kl * dt);
              }
              e_dm[c] = md * dt; e_dT[c] = (Teq - T0) * (1.0 - ex); e_dq[c] = dq * dt; e_dF[c] = eF[c] * (ek - 1.0);
              let fw = eF[c] * exch_weight(1.0 / tauT, kl, dt, !corona);
              e_cond[c] = Nde[c] * fw * md * dt;
              dW -= Nde[c] * fw * md * dt; dH -= Nde[c] * fw * qc * dt;
              Cl += Nde[c] * eF[c] * m * CP_L; ClT += Nde[c] * eF[c] * m * CP_L * eT[c];
            } }
          }
          var m_fav: array<f32, NBM>; var m_dm: array<f32, NBM>; var m_dq: array<f32, NBM>; var m_k: array<f32, NBM>; var m_dt: array<f32, NBM>;
          for (var b = 0; b < nb; b++) {
            let m = max(mm[b], 1e-30); let a = mass_r(m); let vt = v_term(a, rho); let v = max(U - vt, 0.05 * U); let dt = h / v;
            let Td = drop_Teq(a, mq[b], Tg, pv, vt, rho);
            let rr = drop_rates(a, mq[b], Td, Tg, pv, vt, rho); var md = rr.x; var qc = rr.y;
            let mc = evap_cut(md, m, m0[b]); if (md != 0.0) { qc *= mc / md; } md = mc;
            if (md < 0.0 && -md * dt > 0.5 * m) { qc *= (0.5 * m / dt) / -md; md = -0.5 * m / dt; }
            var kk = 0.0; var dq = 0.0;
            if (corona) { let r3 = charge_step(a, mq[b], csz, pol_c, Tg, dt); dq = (r3.x - mq[b]) / dt; kk = 2.0 * (r3.y + r3.z) * csz.E * cunn(a) / (6.0 * PI * MU_G * a) / R; }
            else if (ncol > 0) {
              for (var ci = 0; ci < ncol; ci++) {
                let cv = col[ci]; if (cv.x <= 0.0) { continue; }
                let vr = max(abs(cv.w - v), 1e-3); let stk = RHO_L * 4.0 * a * a * vr / (18.0 * MU_G * 2.0 * cv.y);
                let eimp = (stk / (stk + 0.25)) * (stk / (stk + 0.25));
                let qd = cv.z / pow(2.0 * cv.y, 2.5);
                let kim = qd * qd * a * a * cunn(a) / (3.0 * PI * PI * EPS0 * MU_G * vr);
                kk += cv.x * min(eimp + min(pow(15.0 * PI / 8.0 * kim, 0.4), 1.0), 1.0) * PI * (cv.y + a) * (cv.y + a) * vr;
              }
            }
            m_dm[b] = md * dt; m_dq[b] = dq * dt; m_k[b] = kk; m_dt[b] = dt;
            var Nf = Ndm[b] * mF[b]; if (kk * dt > 1e-7) { Nf = Ndm[b] * mF[b] * (1.0 - exp(-kk * dt)) / (kk * dt); }
            m_fav[b] = Nf;
            dW -= Nf * md * dt; dH -= Nf * qc * dt;
            Cl += Nf * m * CP_L; ClT += Nf * m * CP_L * Td;
          }
          dW += S_v[k] * h / dz; dH += S_h[k] * h / dz;
          let Cg = mair * (CP_A + w * CP_V);
          let dTg = dH / Cg; let dw = dW / mair; var th = 1.0;
          if (Cl > 0.0) {
            let Tl = ClT / Cl; let wl = w_from_pv(psat(Tl));
            if (abs(Tg + dTg - Tl) > abs(Tg - Tl) && (Tg + dTg - Tl) * (Tg - Tl) < 0.0) { th = min(th, (Tl - Tg) / dTg); }
            if (abs(w + dw - wl) > abs(w - wl) && (w + dw - wl) * (w - wl) < 0.0) { th = min(th, (wl - w) / dw); }
          }
          Tg += th * dTg; w = max(w + th * dw, 1e-6);
          if (sprayOn) { for (var c = 0; c < 4; c++) { if (isEnt[c] && eF[c] > 1e-9) {
            let before = eF[c] * em[c];
            em[c] = max(em[c] + th * e_dm[c], 1e-30); eT[c] += th * e_dT[c]; eq[c] += e_dq[c];
            eF[c] = max(eF[c] + e_dF[c], 0.0);
            let loss = Nde[c] * (before - eF[c] * em[c]) + th * e_cond[c];
            eE[c] += e_toE[c] * loss; eW[c] += (1.0 - e_toE[c]) * loss;
          } } }
          for (var b = 0; b < nb; b++) {
            let ek = exp(-m_k[b] * m_dt[b]); let before = mF[b] * mm[b];
            mm[b] = max(mm[b] + th * m_dm[b], 1e-30); mq[b] += m_dq[b]; mF[b] *= ek;
            mD[b] += Ndm[b] * (before - mF[b] * mm[b]) + th * m_fav[b] * m_dm[b];
          }
        }
        let zc_ = (f32(k) + 0.5) * dz;
        if (zb >= zc_ - 1e-7 * L && za <= zc_ + 1e-7 * L) {
          Tprof[k] = Tg; wprof[k] = w;
          var r = 0.0;
          if (sprayOn && !corona) { for (var c = 0; c < 4; c++) { if (isEnt[c]) { let a = mass_r(em[c]); let v = max(U - v_term(a, rho_in), 0.05 * U); r += Nde[c] * eF[c] / (A * v) * eq[c]; } } }
          rqe[k] = r;
        }
      }
    }
    o_w = w; o_top = 0.0; o_wall = 0.0; o_elec = 0.0;
    for (var c = 0; c < 4; c++) { if (isEnt[c]) { o_top += Nde[c] * eF[c] * em[c]; o_wall += eW[c]; o_elec += eE[c]; } }
    o_mistOut = 0.0; o_mistIn = 0.0; o_mistDep = 0.0;
    for (var b = 0; b < nb; b++) { o_mistOut += Ndm[b] * mF[b] * mm[b]; o_mistIn += Ndm[b] * m0[b]; o_mistDep += mD[b]; }
    o_maxRq = maxRq; o_I = Iacc;
  }
  // ---- outlet accounting
  let vapor_in = mair * w_in; let vapor_out = mair * o_w;
  let liq_out = o_mistOut + o_top + f_top;
  let total_in = vapor_in + o_mistIn; let total_out = vapor_out + liq_out;
  let vmax = max_vapor_removal(mair, T_in, w_in, spray_in, X[8]);
  var rho_launch = 0.0;
  for (var c = 0; c < 4; c++) { if (isEnt[c]) { rho_launch += Nde[c] * sp_q[c] / (A * max(U - v_term(sp_a[c], rho_in), 0.05 * U)); } }
  var maxRqf = 0.0; for (var k = 0; k < nz; k++) { maxRqf = max(maxRqf, abs(rqf[k] + rqe[k])); }
  var E_sc = 0.0; if (Qtot > 0.0) { E_sc = max(max(maxRqf, o_maxRq), abs(rho_launch)) * R / (2.0 * EPS0); }
  let re_t = rho_in * U * 2.0 * R / MU_G; let fdr = 0.316 * pow(re_t, -0.25);
  let dp_f = fdr * (L / (2.0 * R)) * 0.5 * rho_in * U * U; let b2 = X[5] * X[5];
  let ndisc = max(1.0, floor(Lc / sp)); let cc = 0.61 + 0.39 * pow(1.0 - b2, 3.0); let kd = pow(1.0 / (cc * (1.0 - b2)) - 1.0, 2.0);
  let shd = min(1.0, sp / (4.0 * X[5] * 2.0 * R));
  var drag = f_drag;
  for (var c = 0; c < 4; c++) { if (isEnt[c]) { drag += Nde[c] * em0[c] * (v0s + U) + Nde[c] * em0[c] * G / max(U - v_term(sp_a[c], rho_in), 0.05 * U) * (L - z_noz); } }
  let dp = dp_f + ndisc * kd * shd * 0.5 * rho_in * U * U + abs(drag) / A;
  let I_c = select(I_corona, o_I, refit);
  let ob = id * ${OUT_STRIDE}u;
  outs[ob + 0u] = select(1.0, total_out / total_in, total_in > 0.0);
  outs[ob + 1u] = (w_in - o_w) / w_in;
  outs[ob + 2u] = liq_out / max(o_mistIn + spray_in, 1e-12);
  outs[ob + 3u] = (total_in - total_out) * 3600.0;
  outs[ob + 4u] = V_c * I_c;
  outs[ob + 5u] = dp * X[0] / FAN_EFF;
  outs[ob + 6u] = min(arc_c, 1.0 - E_sc / E_BD);
  outs[ob + 7u] = I_c * 1e3;
  outs[ob + 8u] = vmax / vapor_in;
  outs[ob + 9u] = dp;
  outs[ob + 10u] = cs.E / 1e5;
  outs[ob + 11u] = cs.V_on / 1e3;
}`;
  }

  class GPUModel {
    static async create(fit, opts = {}) {
      if (!navigator.gpu) throw new Error("WebGPU is not available in this browser");
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
      if (!adapter) throw new Error("No WebGPU adapter");
      const device = await adapter.requestDevice();
      const m = new GPUModel(); m.device = device; m.adapter = adapter;
      m.info = adapter.info || {};
      m.pf = packFit(fit);
      const module = device.createShaderModule({ code: shaderSource(m.pf) });
      const ci = await module.getCompilationInfo();
      const errs = ci.messages.filter(x => x.type === "error");
      if (errs.length) throw new Error("WGSL: " + errs.map(e => `${e.lineNum}:${e.linePos} ${e.message}`).join("; "));
      m.pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } });
      m.fitBuf = device.createBuffer({ size: Math.ceil(m.pf.data.byteLength / 16) * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(m.fitBuf, 0, m.pf.data);
      m.uni = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      m.cap = 0;
      return m;
    }
    ensure(n) {
      if (n <= this.cap) return;
      const d = this.device;
      for (const b of [this.inBuf, this.outBuf, this.readBuf]) if (b) b.destroy();
      this.cap = n;
      this.inBuf = d.createBuffer({ size: n * IN_STRIDE * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      this.outBuf = d.createBuffer({ size: n * OUT_STRIDE * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      this.readBuf = d.createBuffer({ size: n * OUT_STRIDE * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      this.bind = d.createBindGroup({ layout: this.pipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: this.inBuf } }, { binding: 1, resource: { buffer: this.outBuf } },
        { binding: 2, resource: { buffer: this.fitBuf } }, { binding: 3, resource: { buffer: this.uni } }] });
    }
    // designs: Float32Array (n * 20); opt: {nb, nz, outer, corona}
    async run(designs, n, opt = {}) {
      this.ensure(n);
      const d = this.device, u = new ArrayBuffer(32), dv = new DataView(u);
      dv.setUint32(0, n, true); dv.setInt32(4, opt.nb || 4, true); dv.setInt32(8, Math.min(opt.nz || 40, 64), true); dv.setInt32(12, opt.outer || 3, true);
      dv.setInt32(16, opt.corona === "rom" ? 1 : 0, true); dv.setInt32(20, opt.nsub_disp || 128, true); dv.setFloat32(24, opt.spray_dt_frac || 0.5, true);
      d.queue.writeBuffer(this.uni, 0, u);
      d.queue.writeBuffer(this.inBuf, 0, designs.buffer, designs.byteOffset, n * IN_STRIDE * 4);
      const enc = d.createCommandEncoder(), pass = enc.beginComputePass();
      pass.setPipeline(this.pipeline); pass.setBindGroup(0, this.bind); pass.dispatchWorkgroups(Math.ceil(n / 64)); pass.end();
      enc.copyBufferToBuffer(this.outBuf, 0, this.readBuf, 0, n * OUT_STRIDE * 4);
      d.queue.submit([enc.finish()]);
      await this.readBuf.mapAsync(GPUMapMode.READ, 0, n * OUT_STRIDE * 4);
      const out = new Float32Array(this.readBuf.getMappedRange(0, n * OUT_STRIDE * 4).slice(0));
      this.readBuf.unmap();
      return out;
    }
  }
  function packDesign(arr, i, x) { const o = i * IN_STRIDE; for (let k = 0; k < IN.length; k++) arr[o + k] = x[IN[k]]; }
  const api = { GPUModel, shaderSource, packFit, packDesign, IN, OUT, IN_STRIDE, OUT_STRIDE };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.WCGPU = api;
})(typeof self !== "undefined" ? self : this);
