// Watercap explorer model: a GPU-portable rewrite of Vega's watercap_oracle.py
// (the "fast model"), one design per call, fixed-size state, no closures over
// growable arrays, so the WGSL port is line-for-line.
//
// Differences from the Python ROM, all deliberate:
//  - spray is given as drop diameter + charge (fraction of Rayleigh), not an
//    electrospray operating point (the explorer's question is "what spray",
//    and an atomizer + charge-to-mass is the hardware the findings point to);
//    main drops + 15% satellites at half radius, q/4, exactly as the ROM
//  - fixed-step z-march instead of LSODA: drop temperatures relax
//    exponentially (exact for the linearized exchange), the gas is
//    sub-stepped by its stiffness, and each sub-step is capped at the joint
//    gas+liquid equilibrium, so it cannot overshoot however stiff the cell
//  - corona = "refit" (onset, current, wall field and ion density fitted to
//    stonelab.ehd resolved solves) or "rom" (the ROM's coaxial equivalent,
//    kept for checking this port against the Python ROM)
(function (root) {
  "use strict";
  const P_ATM = 101325.0, R_D = 287.05, R_V = 461.5, R_U = 8.314, M_W = 0.018015;
  const CP_A = 1006.0, CP_V = 1860.0, CP_L = 4186.0, RHO_L = 998.0, EPS0 = 8.854e-12, EPS_R_W = 80.0;
  const E_CHG = 1.602e-19, K_B = 1.381e-23, G = 9.81, MU_G = 1.81e-5, K_G = 0.026, PR = 0.71, SC = 0.60;
  const DISC_EDGE_RADIUS_M = 1.5e-4, SPRAY_V0 = 2.0, FAN_EFF = 0.6, E_BREAKDOWN = 2.4e6;
  const SPARK_E_INF = { "1": 4.5, "-1": 8.0 }, MOB = { "1": 1.4e-4, "-1": 2.1e-4 };
  const PI = Math.PI;
  const clamp = (x, a, b) => Math.min(b, Math.max(a, x));

  // ---------------- thermo ----------------
  function psat(T) { T = clamp(T, -60, 150); return 611.21 * Math.exp((18.678 - T / 234.5) * (T / (257.14 + T))); }
  function dpsat(T) { T = clamp(T, -60, 150); const u = T / (257.14 + T); return psat(T) * (-u / 234.5 + (18.678 - T / 234.5) * 257.14 / ((257.14 + T) * (257.14 + T))); }
  const w_from_pv = pv => 0.622 * pv / (P_ATM - pv);
  const pv_from_w = w => w * P_ATM / (0.622 + w);
  const h_fg = T => 2.501e6 - 2370.0 * T;
  const sigma = T => 0.0756 - 1.4e-4 * T;
  const Dv = T => 2.11e-5 * Math.pow((T + 273.15) / 273.15, 1.94);
  function rho_gas(T, w) { const pv = pv_from_w(w), Tk = T + 273.15; return (P_ATM - pv) / (R_D * Tk) + pv / (R_V * Tk); }
  function kt_factor(a, q, T) {
    const term = 2 * sigma(T) / a - (1 - 1 / EPS_R_W) * q * q / (32 * PI * PI * EPS0 * a * a * a * a);
    return Math.exp(clamp(M_W / (RHO_L * R_U * (T + 273.15)) * term, -5, 5));
  }
  const rayleigh = (a, T) => 8 * PI * Math.sqrt(EPS0 * sigma(T) * a * a * a);
  function v_term(a, rho) { let v = 2 * RHO_L * G * a * a / (9 * MU_G); for (let i = 0; i < 8; i++) { const re = rho * v * 2 * a / MU_G; v = 2 * RHO_L * G * a * a / (9 * MU_G * (1 + 0.15 * Math.pow(re, 0.687))); } return v; }
  function cunn(a) { const kn = 0.066e-6 / a; return 1 + kn * (1.257 + 0.4 * Math.exp(-1.1 / kn)); }
  function nu_sh(a, vrel, rho) { const s = 0.6 * Math.sqrt(rho * Math.abs(vrel) * 2 * a / MU_G); return [2 + s * Math.cbrt(PR), 2 + s * Math.cbrt(SC)]; }
  function drop_Teq(a, q, Tg, pv, vrel, rho) {
    const [nu, sh] = nu_sh(a, vrel, rho), D = Dv(Tg), rvi = pv / (R_V * (Tg + 273.15));
    let Td = Tg - 0.5;
    for (let i = 0; i < 6; i++) {
      const Tk = Td + 273.15, kt = kt_factor(a, q, Td), rvs = psat(Td) * kt / (R_V * Tk);
      const drho = dpsat(Td) * kt / (R_V * Tk) - rvs / Tk;
      const f = nu * K_G * (Tg - Td) + h_fg(Td) * sh * D * (rvi - rvs), df = -nu * K_G - h_fg(Td) * sh * D * drho;
      Td -= f / df;
    }
    return Td;
  }
  function drop_rates(a, q, Td, Tg, pv, vrel, rho) {
    const [nu, sh] = nu_sh(a, vrel, rho), D = Dv(Tg);
    const rvi = pv / (R_V * (Tg + 273.15)), rvs = psat(Td) * kt_factor(a, q, Td) / (R_V * (Td + 273.15));
    return [2 * PI * a * sh * D * (rvi - rvs), 2 * PI * a * nu * K_G * (Tg - Td)];
  }
  function evap_cut(md, m, mref) { if (md >= 0) return md; const r = m / (m + 1e-6 * mref); return md * r * r; }
  const peek = r => 3.1e6 * (1 + 0.0308 / Math.sqrt(r));
  function spark_V(gap, pol) { const gc = gap * 100; return 1e3 * SPARK_E_INF[pol] * (1 + 1 / Math.sqrt(gc)) * gc; }

  // Acklam's inverse normal CDF (|err| < 1.2e-9)
  function ppf(p) {
    const a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.383577518672690e2, -3.066479806614716e1, 2.506628277459239];
    const b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1];
    const c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
    const d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416];
    if (p < 0.02425) { const q = Math.sqrt(-2 * Math.log(p)); return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
    if (p > 1 - 0.02425) return -ppf(1 - p);
    const q = p - 0.5, r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  }

  // ---------------- corona ----------------
  function corona_rom(V, R, rd, s, pol) {
    const gap = R - rd, shield = 1 + Math.exp(-2 * s / Math.max(gap, 1e-6));
    const V_on = peek(DISC_EDGE_RADIUS_M) * DISC_EDGE_RADIUS_M * Math.log(2 * gap / DISC_EDGE_RADIUS_M) * shield;
    const mu = MOB[pol], lnr = Math.log(R / Math.max(rd, 1e-4));
    if (V <= V_on) return { V_on, I_per_m: 0, E_wall: V / (R * lnr), N_ion: 0, mu };
    const I = 8 * PI * EPS0 * mu * V * (V - V_on) / (R * R * lnr);
    const E_lap = V_on / (R * lnr), E_wall = Math.sqrt(E_lap * E_lap + I / (2 * PI * EPS0 * mu));
    return { V_on, I_per_m: I, E_wall, N_ion: I / (2 * PI * R) / (E_CHG * mu * E_wall), mu };
  }
  // trilinear lookup in a fitted table; t is clamped to [-ex, 1+ex] so a
  // table with ex > 0 extrapolates linearly by up to ex of an edge cell
  function lattice(tb, v0, v1, v2) {
    const v = [v0, v1, v2], ii = [0, 0, 0], tt = [0, 0, 0], ex = tb.extrap;
    for (let d = 0; d < 3; d++) {
      const g = tb.axes[d]; let i = 0; while (i < g.length - 2 && v[d] > g[i + 1]) i++;
      ii[d] = i; tt[d] = clamp((v[d] - g[i]) / (g[i + 1] - g[i]), -ex, 1 + ex);
    }
    const n1 = tb.axes[1].length, n2 = tb.axes[2].length, T = tb.vals;
    let out = 0;
    for (let a = 0; a < 2; a++) for (let b = 0; b < 2; b++) for (let c = 0; c < 2; c++)
      out += (a ? tt[0] : 1 - tt[0]) * (b ? tt[1] : 1 - tt[1]) * (c ? tt[2] : 1 - tt[2]) * T[((ii[0] + a) * n1 + ii[1] + b) * n2 + ii[2] + c];
    return out;
  }
  // Corona refit (stonelab.ehd): onset and wall field per volt from Laplace
  // solves; current per disc period I = J (V/V_on)(V/V_on - 1) from Kaptsov
  // solves (Townsend form holds to <2% between 1.2x and 1.6x onset); mean
  // gap field and ion density for charging/drift from the same solves.
  function corona_refit(V, R, rd, s, pol, fit, dVon = 0) {
    const fr = rd / R, mu = MOB[pol], t = fit.tables, lR = Math.log(R), lS = Math.log(s);
    const V_on0 = Math.exp(lattice(t.lnVon, lR, fr, lS)), Ewpv = Math.exp(lattice(t.lnEwall, lR, fr, lS));
    const J = Math.exp(lattice(t.lnJ, lR, Math.log(fr), lS)) * (mu / fit.mu_ref);
    return corona_eval({ V_on0, Ewpv, J, R, s, mu, cE: fit.cE, cN: fit.cN, kq: fit.kq || 1 }, V, dVon);
  }
  // Quenching: drop space charge rho_p of the ions' sign raises the voltage
  // the emitter needs by ~ rho_p R^2 / (4 eps0) (uniform space charge in a
  // grounded tube); dVon is that shift, computed cell by cell in the march.
  function corona_eval(b, V, dVon) {
    const V_on = b.V_on0 + dVon, R = b.R, mu = b.mu;
    if (V <= V_on) return { V_on, V_on0: b.V_on0, I_per_m: 0, E_wall: V * b.Ewpv, E_charge: V * b.Ewpv, N_ion: 0, mu, base: b };
    const xr = V / V_on, I = b.J * xr * (xr - 1) / b.s;
    const E_lap = V * b.Ewpv, E = b.cE * Math.sqrt(E_lap * E_lap + I / (2 * PI * EPS0 * mu));
    return { V_on, V_on0: b.V_on0, I_per_m: I, E_wall: E, E_charge: b.kq * E, N_ion: b.cN * I / (2 * PI * R) / (E_CHG * mu * E), mu, base: b };
  }
  // Exact charge update over dt (charging time tau can be << dt): field
  // charging integrated in closed form (Pauthenier for 0 <= q_rel < q_s; the
  // linear opposite-sign branch of charge_rate relaxes exponentially until
  // q_rel = 0), diffusion charging explicit but capped at one kT/e of
  // surface potential per step. Returns [q_new, step-average q].
  // Exact charge update over dt (charging time tau can be << dt): field
  // charging in closed form (Pauthenier for 0 <= q_rel < q_s; the linear
  // opposite-sign branch of charge_rate relaxes exponentially until q_rel = 0),
  // diffusion charging explicit but capped at one kT/e of surface potential.
  // Returns [q_new, avg q_rel while >= 0, avg |q_rel| while < 0] (both
  // averaged over the whole step): a drop that flips sign mid-step drifts
  // first to the electrode and then to the wall, and a signed average would
  // cancel the two.
  function charge_step(a, q, cs, pol, T, dt) {
    const qr0 = pol * q;
    if (cs.N_ion <= 0 || dt <= 0) return [q, Math.max(qr0, 0), Math.max(-qr0, 0)];
    const p = 3 * EPS_R_W / (EPS_R_W + 2), qs = 4 * PI * EPS0 * p * (cs.E_charge || cs.E_wall) * a * a;
    const tau = 4 * EPS0 / (cs.N_ion * E_CHG * cs.mu);
    const Tk = T + 273.15, cbar = Math.sqrt(8 * K_B * Tk / (PI * 4.8e-26)), cy = E_CHG / (4 * PI * EPS0 * a * K_B * Tk);
    const D0 = PI * a * a * cs.N_ion * E_CHG * cbar;
    let qr = qr0, t = dt, accP = 0, accN = 0;
    if (qr < 0) {
      // opposite sign: field branch (qs/tau)(1 + 4|q|/qs) plus attractive
      // diffusion D0 (1 + y), y = |q| cy (bridges y/(1-e^-y) between its
      // limits) -- linear in q, so it relaxes exponentially toward qstar > 0
      const lam = 4 / tau + D0 * cy, qst = (qs / tau + D0) / lam;
      const t0 = Math.log((qst - qr) / qst) / lam, tt = Math.min(t, t0), e = Math.exp(-lam * tt);
      accN -= qst * tt + (qr - qst) * (1 - e) / lam;
      qr = tt < t0 ? qst + (qr - qst) * e : 0; t -= tt;
    }
    if (t > 0) {
      if (qr < qs) {
        const t1 = tau * qr / (qs - qr), u = t / (t1 + tau);
        // qs [t - tau ln(1 + u)], series for small u (cancellation)
        accP += qs * (u < 1e-3 ? t * t1 / (t1 + tau) + tau * u * u * (0.5 - u / 3) : t - tau * Math.log(1 + u));
        qr = qs * (t1 + t) / (t1 + t + tau);
      } else accP += qr * t;
    }
    // repulsive diffusion charging (q_rel >= 0, self-limiting): explicit, capped
    const dqd = t > 0 ? Math.min(D0 * Math.exp(-Math.min(qr * cy, 700)) * t, 1 / cy) : 0;
    qr += dqd;
    return [pol * qr, Math.max(accP / dt + (qr >= 0 ? 0.5 * dqd : 0), 0), Math.max(accN / dt, 0)];
  }
  function scav(a_m, v_m, col, nc) {  // col: flat [n, a, q, v] * nc
    let tot = 0;
    for (let c = 0; c < nc; c++) {
      const n = col[4 * c], ac = col[4 * c + 1], qc = col[4 * c + 2], vc = col[4 * c + 3];
      if (n <= 0) continue;
      const vr = Math.max(Math.abs(vc - v_m), 1e-3), stk = RHO_L * 4 * a_m * a_m * vr / (18 * MU_G * 2 * ac);
      const eimp = (stk / (stk + 0.25)) ** 2;
      const kim = qc * qc * a_m * a_m * cunn(a_m) / (3 * PI * PI * EPS0 * MU_G * Math.pow(2 * ac, 5) * vr);
      tot += n * Math.min(eimp + Math.min(Math.pow(15 * PI / 8 * kim, 0.4), 1), 1) * PI * (ac + a_m) * (ac + a_m) * vr;
    }
    return tot;
  }
  function max_vapor_removal(mair, Tin, win, ms, Ts) {
    if (ms <= 0 || Ts >= Tin) return 0;
    let lo = Ts, hi = Tin;
    for (let i = 0; i < 50; i++) {
      const mid = 0.5 * (lo + hi), wf = Math.min(w_from_pv(psat(mid)), win);
      const rel = mair * (CP_A + win * CP_V) * (Tin - mid) + h_fg(mid) * mair * (win - wf);
      if (rel - ms * CP_L * (mid - Ts) > 0) lo = mid; else hi = mid;
    }
    return Math.max(mair * (win - w_from_pv(psat(0.5 * (lo + hi)))), 0);
  }
  // joint equilibrium of the gas stream with a liquid stream (heat capacity
  // flow Cl at mean T Tl, mass flow Lq available to evaporate): (T_f, w_f)
  function joint_eq(mair, Tg, w, Cl, Tl, Lq) {
    let lo = Math.min(Tg, Tl) - 40, hi = Math.max(Tg, Tl);
    for (let i = 0; i < 40; i++) {
      const mid = 0.5 * (lo + hi), wf = Math.min(w_from_pv(psat(mid)), w + Lq / mair);
      const f = mair * (CP_A + w * CP_V) * (Tg - mid) + h_fg(mid) * mair * (w - wf) - Cl * (mid - Tl);
      if (f > 0) lo = mid; else hi = mid;
    }
    const Tf = 0.5 * (lo + hi);
    return [Tf, Math.min(w_from_pv(psat(Tf)), w + Lq / mair)];
  }

  // g(c) = e^c E1(c), Abramowitz & Stegun 5.1.53 / 5.1.56
  function eE1(c) {
    if (c <= 1) return Math.exp(c) * (-Math.log(c) - 0.57721566 + c * (0.99999193 + c * (-0.24991055 + c * (0.05519968 + c * (-0.00976004 + c * 0.00107857)))));
    return (c * c + 2.334733 * c + 0.250621) / (c * (c * c + 3.330657 * c + 1.681534));
  }
  // Fraction of the drops' exchange that happens before they are lost:
  // int F(t) e^(-ke t) dt / int e^(-ke t) dt over the step, F(0) = 1, with
  // F = 1/(1 + kl t) (Coulomb dispersion, hyperbolic) or e^(-kl t) (fixed field).
  function exch_weight(ke, kl, dt, hyper) {
    if (kl * dt < 1e-9) return 1;
    const den = ke * dt > 1e-9 ? (1 - Math.exp(-ke * dt)) / ke : dt;
    if (!hyper) return ((ke + kl) * dt > 1e-9 ? (1 - Math.exp(-(ke + kl) * dt)) / (ke + kl) : dt) / den;
    if (ke * dt < 1e-6) return Math.log(1 + kl * dt) / (kl * dt);
    const c0 = ke / kl, c1 = c0 * (1 + kl * dt);
    return Math.min(1, Math.max(0, (eE1(c0) - Math.exp(-ke * dt) * eE1(c1)) / kl / den));
  }

  const NB_MAX = 8, NS = 4, NZ_MAX = 200;

  function simulate(x, opt = {}) {
    const nb = opt.nb || 4, nz = opt.nz || 60, outer = opt.outer || 4, dtf = opt.spray_dt_frac || 0.5;
    const pol_c = x.corona_polarity >= 0 ? 1 : -1, pol_s = x.spray_polarity_rel >= 0 ? pol_c : -pol_c;
    const R = x.tube_diameter_m / 2, A = PI * R * R, Ls = x.spray_section_length_m, Lc = x.corona_section_length_m, L = Ls + Lc;
    const z_noz = x.nozzle_height_frac * Ls, rd = x.disc_diameter_frac * R;
    const T_in = x.inlet_T_C, w_in = w_from_pv(x.inlet_RH * psat(T_in)), rho_in = rho_gas(T_in, w_in);
    const mair = rho_in * x.air_flow_m3_s / (1 + w_in), U = x.air_flow_m3_s / A;
    const V_c = x.corona_voltage_kV * 1e3;
    const cs = opt.corona === "rom" || !opt.fit ? corona_rom(V_c, R, rd, x.disc_spacing_m, pol_c) : corona_refit(V_c, R, rd, x.disc_spacing_m, pol_c, opt.fit);
    const I_corona = cs.I_per_m * Lc;
    const arc_c = 1 - V_c / spark_V(R - rd, pol_c);

    // spray classes: main 85% of volume at a, satellites 15% at a/2 with q/4
    const Qtot = x.spray_flow_L_min / 60000, a_s = x.spray_d_um * 0.5e-6, q_s = pol_s * x.spray_q_frac * rayleigh(a_s, x.spray_T_C);
    // lognormal spray (4 equal-volume bins, charge a fixed fraction of each
    // bin's Rayleigh limit, as the ROM's atomizer override) when a spread is
    // given; else main drops 85% at a + satellites 15% at a/2, q/4
    const gsd = x.spray_gsd || 1, v0s = x.spray_v0 || SPRAY_V0;
    let sp_a, sp_q, sp_Q;
    if (gsd > 1.0001) {
      sp_a = [0, 1, 2, 3].map(c => a_s * Math.exp(Math.log(gsd) * ppf((c + 0.5) / 4)));
      sp_q = sp_a.map(a => pol_s * x.spray_q_frac * rayleigh(a, x.spray_T_C)); sp_Q = [0.25, 0.25, 0.25, 0.25].map(f => f * Qtot);
    } else { sp_a = [a_s, a_s / 2, a_s, a_s]; sp_q = [q_s, q_s / 4, 0, 0]; sp_Q = [0.85 * Qtot, 0.15 * Qtot, 0, 0]; }
    const spray_in = RHO_L * Qtot;
    const isEnt = [false, false, false, false], isFall = [false, false, false, false];
    for (let c = 0; c < NS; c++) if (sp_Q[c] > 0) { if (v_term(sp_a[c], rho_in) < 0.8 * U) isEnt[c] = true; else isFall[c] = true; }
    const anyFall = isFall.some(Boolean), anyEnt = isEnt.some(Boolean);
    const em0 = sp_a.map(a => RHO_L * 4 / 3 * PI * a * a * a), Nde = [0, 1, 2, 3].map(c => RHO_L * sp_Q[c] / em0[c]);

    // mist bins
    const m0 = new Float64Array(NB_MAX), Ndm = new Float64Array(NB_MAX), am0 = new Float64Array(NB_MAX);
    for (let b = 0; b < nb; b++) {
      const d = x.mist_mmd_um * 1e-6 * Math.exp(Math.log(Math.max(x.mist_gsd, 1 + 1e-6)) * ppf((b + 0.5) / nb));
      am0[b] = d / 2; m0[b] = RHO_L * PI * d * d * d / 6; Ndm[b] = (x.inlet_lwc_g_m3 * 1e-3 / nb) / m0[b] * x.air_flow_m3_s;
    }
    const dz = L / nz;
    const Tprof = new Float64Array(nz).fill(T_in), wprof = new Float64Array(nz).fill(w_in);
    const S_v = new Float64Array(nz), S_h = new Float64Array(nz), rqf = new Float64Array(nz), rqe = new Float64Array(nz);
    const fdens = [0, 1, 2, 3].map(() => new Float64Array(nz));
    let fst = { top: 0, bottom: 0, wall: 0, elec: 0, drag: 0 };
    let st;  // march result
    const nIter = anyFall ? outer : 1;
    for (let it = 0; it < nIter; it++) {
      if (anyFall) {
        const r = trace_falling(), relax = it ? 0.5 : 1;
        for (let k = 0; k < nz; k++) { S_v[k] = relax * r.S_v[k] + (1 - relax) * S_v[k]; S_h[k] = relax * r.S_h[k] + (1 - relax) * S_h[k]; rqf[k] = relax * r.rq[k] + (1 - relax) * rqf[k]; }
        fst = r.st;
      }
      st = march();
      for (let k = 0; k < nz; k++) { Tprof[k] = st.Tp[k]; wprof[k] = st.wp[k]; rqe[k] = st.rqe[k]; }
    }

    function trace_falling() {
      const Sv = new Float64Array(nz), Sh = new Float64Array(nz), rq = new Float64Array(nz);
      const s = { top: 0, bottom: 0, wall: 0, elec: 0, drag: 0 };
      for (let c = 0; c < NS; c++) {
        fdens[c].fill(0);
        if (!isFall[c]) continue;
        const Nd = Nde[c];
        let z = z_noz, v = -v0s, a = sp_a[c], q = sp_q[c], f = 1, Td = Tprof[Math.min(Math.floor(z / dz), nz - 1)], steps = 0;
        while (z >= 0 && z < L && f > 1e-6 && steps < (opt.fall_steps || 20000)) {
          const k = Math.min(Math.floor(z / dz), nz - 1), Tg = Tprof[k], wg = wprof[k];
          const rho = rho_gas(Tg, wg), pv = pv_from_w(wg), vrel = U - v, re = rho * Math.abs(vrel) * 2 * a / MU_G;
          const taup = RHO_L * 4 * a * a / (18 * MU_G * (1 + 0.15 * Math.pow(re, 0.687))), m = RHO_L * 4 / 3 * PI * a * a * a;
          const Teq = drop_Teq(a, q, Tg, pv, vrel, rho), [nu, sh] = nu_sh(a, vrel, rho);
          const dpdT = dpsat(Td) / (R_V * (Td + 273.15));
          const tauT = Math.max(m * CP_L / (2 * PI * a * (nu * K_G + h_fg(Td) * sh * Dv(Tg) * dpdT)), 1e-9);
          const Zc = cunn(a) / (6 * PI * MU_G * a);
          // step from the cell transit, with the speed floored at 5% of U: a drop
          // hovering at v ~ 0 would otherwise take one step of many seconds
          let dt = dtf * dz / Math.max(Math.abs(v), 0.05 * U), dq = 0, toE = 0, kl;
          if (z >= Ls) {
            const [qn, qP, qN] = charge_step(a, q, cs, pol_c, Tg, dt); dq = (qn - q) / dt;
            kl = 2 * (qP + qN) * cs.E_wall * Zc / R; toE = qN / Math.max(qP + qN, 1e-300);
          } else kl = Math.abs(q) * Zc * Math.abs(rqf[k] + rqe[k]) / EPS0;
          let eT = Math.exp(-dt / tauT), Tavg = Teq + (Td - Teq) * tauT / dt * (1 - eT);
          let [md, qc] = drop_rates(a, q, Tavg, Tg, pv, vrel, rho);
          if (Math.abs(md) * dt > 0.2 * m) { dt = 0.2 * m / Math.abs(md); eT = Math.exp(-dt / tauT); Tavg = Teq + (Td - Teq) * tauT / dt * (1 - eT); [md, qc] = drop_rates(a, q, Tavg, Tg, pv, vrel, rho); }
          const Tnew = Teq + (Td - Teq) * eT, ek = Math.exp(-kl * dt), fn = f * ek, favg = kl * dt > 1e-12 ? f * (1 - ek) / (kl * dt) : f;
          const veq = U - G * taup, ep = Math.exp(-dt / taup), vn = veq + (v - veq) * ep, dzp = veq * dt + (v - veq) * taup * (1 - ep);
          const mn = Math.max(m + md * dt, 1e-30), cond = md * Nd * favg * dt, lost = Nd * (f * m - fn * mn) + cond;
          s.elec += toE * lost; s.wall += (1 - toE) * lost;
          Sv[k] -= cond; Sh[k] -= qc * Nd * favg * dt; s.drag += Nd * favg * m * (vn - v + G * dt);
          fdens[c][k] += Nd * favg * dt; rq[k] += q * Nd * favg * dt;
          a = Math.cbrt(mn / (RHO_L * 4 / 3 * PI)); q += dq * dt;
          const qR = rayleigh(a, Tavg); if (Math.abs(q) > qR) q = Math.sign(q) * qR;
          Td = Tnew; v = vn; f = fn; z += dzp; steps++;
        }
        const me = RHO_L * 4 / 3 * PI * a * a * a;
        if (z >= L) s.top += Nd * f * me; else if (z < 0) s.bottom += Nd * f * me; else s.wall += Nd * f * me;
        for (let k = 0; k < nz; k++) fdens[c][k] /= A * dz;
      }
      for (let k = 0; k < nz; k++) rq[k] /= A * dz;
      return { S_v: Sv, S_h: Sh, rq, st: s };
    }

    function march() {
      let Tg = T_in, w = w_in;
      const mm = Float64Array.from(m0), mq = new Float64Array(NB_MAX), mF = new Float64Array(NB_MAX).fill(1), mD = new Float64Array(NB_MAX);
      const em = Float64Array.from(em0), eT = new Float64Array(NS).fill(x.spray_T_C), eq = Float64Array.from(sp_q), eF = new Float64Array(NS).fill(1);
      const eW = new Float64Array(NS), eE = new Float64Array(NS);
      const Tp = new Float64Array(nz), wp = new Float64Array(nz), rqo = new Float64Array(nz);
      const lqm = new Float64Array(nz), lqs = new Float64Array(nz);
      const col = new Float64Array(4 * (2 * NS));
      let maxRq = 0, nCap = 0, Iacc = 0;
      // breakpoints: uniform cell edges + nozzle + section boundary
      for (let k = 0; k < nz; k++) {
        const z0 = k * dz, z1 = (k + 1) * dz, cuts = [z0];
        if (z_noz > z0 && z_noz < z1) cuts.push(z_noz);
        if (Ls > z0 && Ls < z1) cuts.push(Ls);
        cuts.sort((p, q) => p - q); cuts.push(z1);
        for (let sgi = 0; sgi + 1 < cuts.length; sgi++) {
          const za = cuts[sgi], zb = cuts[sgi + 1], zm = 0.5 * (za + zb), hseg = zb - za;
          if (hseg <= 1e-12 * L) continue;  // coincident breakpoints (nozzle at the section boundary)
          const corona = zm >= Ls, sprayOn = anyEnt && zm >= z_noz;
          let csz = cs;
          const qk = opt.quench ?? (opt.fit && opt.fit.quench) ?? 0;
          if (corona && cs.base && qk > 0) {
            const rho = rho_gas(Tg, w);
            let rp = 0;
            for (let b = 0; b < nb; b++) { const a = Math.cbrt(Math.max(mm[b], 1e-30) / (RHO_L * 4 / 3 * PI)), v = Math.max(U - v_term(a, rho), 0.05 * U); rp += Ndm[b] * mF[b] / (A * v) * mq[b]; }
            if (sprayOn) for (let c = 0; c < NS; c++) if (isEnt[c]) { const a = Math.cbrt(Math.max(em[c], 1e-30) / (RHO_L * 4 / 3 * PI)), v = Math.max(U - v_term(a, rho), 0.05 * U); rp += Nde[c] * eF[c] / (A * v) * eq[c]; }
            csz = corona_eval(cs.base, V_c, qk * Math.max(0, pol_c * rp) * R * R / (4 * EPS0));
          }
          if (corona) Iacc += csz.I_per_m * hseg;
          // stiffness estimate -> number of sub-steps
          let lam = 0, lamD = 0;
          {
            const rho = rho_gas(Tg, w), Cg = mair * (CP_A + w * CP_V), dvw = 0.622 * P_ATM / ((0.622 + w) * (0.622 + w)) / (R_V * (Tg + 273.15));
            for (let b = 0; b < nb; b++) {
              const a = Math.cbrt(Math.max(mm[b], 1e-30) / (RHO_L * 4 / 3 * PI)), vt = v_term(a, rho), v = Math.max(U - vt, 0.05 * U), [nu, sh] = nu_sh(a, vt, rho);
              const Nf = Ndm[b] * mF[b] / v;
              lam += Nf * 2 * PI * a * (nu * K_G / Cg + sh * Dv(Tg) * dvw / mair);
            }
            if (sprayOn) for (let c = 0; c < NS; c++) if (isEnt[c]) {
              const a = Math.cbrt(Math.max(em[c], 1e-30) / (RHO_L * 4 / 3 * PI)), vt = v_term(a, rho), v = Math.max(U - vt, 0.05 * U), [nu, sh] = nu_sh(a, vt, rho);
              const Nf = Nde[c] * eF[c] / v;
              lam += Nf * 2 * PI * a * (nu * K_G / Cg + sh * Dv(Tg) * dvw / mair);
              if (!corona) lamD += Math.abs(eq[c]) * cunn(a) / (6 * PI * MU_G * a) * Math.abs(rqf[k] + Nde[c] * eF[c] / (A * v) * eq[c]) / EPS0 / v;
            }
          }
          const nsub = Math.max(clamp(Math.ceil(lam / 0.7 * hseg), 1, 32), clamp(Math.ceil(lamD / 0.5 * hseg), 1, opt.nsub_disp || 128)), h = hseg / nsub;
          for (let ss = 0; ss < nsub; ss++) {
            const rho = rho_gas(Tg, w), pv = pv_from_w(w);
            let rq = rqf[k], dW = 0, dH = 0;                 // gas: kg vapor and J into gas over the sub-step
            let ncol = 0;
            if (!corona) for (let c = 0; c < NS; c++) if (isFall[c] && fdens[c][k] > 0) {
              col[4 * ncol] = fdens[c][k]; col[4 * ncol + 1] = sp_a[c]; col[4 * ncol + 2] = sp_q[c]; col[4 * ncol + 3] = U - v_term(sp_a[c], rho_gas(Tprof[0], wprof[0])); ncol++;
            }
            // entrained spray increments (explicit, drop T exponential)
            const e_cond = [0, 0, 0, 0], e_dm = [0, 0, 0, 0], e_dT = [0, 0, 0, 0], e_dq = [0, 0, 0, 0], e_dF = [0, 0, 0, 0], e_toE = [0, 0, 0, 0];
            let Cl = 0, ClT = 0, Lq = 0;
            if (sprayOn) {
              let rqE = 0;
              for (let c = 0; c < NS; c++) if (isEnt[c] && eF[c] > 1e-9) {
                const m = Math.max(em[c], 1e-30), a = Math.cbrt(m / (RHO_L * 4 / 3 * PI)), qR = rayleigh(a, eT[c]);
                const q = Math.abs(eq[c]) > qR ? Math.sign(eq[c]) * qR : eq[c], vt = v_term(a, rho), v = Math.max(U - vt, 0.05 * U);
                rqE += Nde[c] * eF[c] / (A * v) * q;
              }
              rq += rqE; if (!corona) maxRq = Math.max(maxRq, Math.abs(rq));
              for (let c = 0; c < NS; c++) if (isEnt[c] && eF[c] > 1e-9) {
                const m = Math.max(em[c], 1e-30), a = Math.cbrt(m / (RHO_L * 4 / 3 * PI)), qR = rayleigh(a, eT[c]);
                const q = Math.abs(eq[c]) > qR ? Math.sign(eq[c]) * qR : eq[c], vt = v_term(a, rho), v = Math.max(U - vt, 0.05 * U), dt = h / v;
                // secant exponential integrator: the heat input H(T) = qconv + h_fg mdot
                // is zero at Teq, so the secant through (T0, H0) and (Teq, 0) gives a
                // relaxation time that is exact at both ends; mdot and qconv are
                // interpolated on the same secant, so drop and gas see the same
                // fluxes and energy is conserved however far the drop moves.
                const T0 = eT[c], Teq = drop_Teq(a, q, Tg, pv, vt, rho);
                let [md0, qc0] = drop_rates(a, q, T0, Tg, pv, vt, rho);
                const [mde, qce] = drop_rates(a, q, Teq, Tg, pv, vt, rho);
                const H0 = qc0 + h_fg(T0) * md0;
                let tauT;
                if (Math.abs(T0 - Teq) > 1e-4) tauT = m * CP_L * (T0 - Teq) / Math.max(-H0 * Math.sign(T0 - Teq), 1e-30) * Math.sign(T0 - Teq);
                else { const [nu, sh] = nu_sh(a, vt, rho); tauT = m * CP_L / (2 * PI * a * (nu * K_G + h_fg(T0) * sh * Dv(Tg) * dpsat(T0) / (R_V * (T0 + 273.15)))); }
                if (!(tauT > 0)) tauT = 1e-9;
                const ex = Math.exp(-dt / tauT), u = Math.abs(T0 - Teq) > 1e-4 ? tauT / dt * (1 - ex) : 1;  // (Tavg-Teq)/(T0-Teq)
                let md = mde + (md0 - mde) * u, qc = qce + (qc0 - qce) * u;
                const mcut = evap_cut(md, m, em0[c]); if (md !== 0) qc *= mcut / md; md = mcut;
                if (md < 0 && -md * dt > 0.5 * m) { qc *= (0.5 * m / dt) / -md; md = -0.5 * m / dt; }
                const Zc = cunn(a) / (6 * PI * MU_G * a);
                let kl, dq = 0, ek, favg;
                if (corona) {
                  const [qn, qP, qN] = charge_step(a, q, csz, pol_c, Tg, dt); dq = (qn - q) / dt;
                  kl = 2 * (qP + qN) * csz.E_wall * Zc / R; e_toE[c] = qN / Math.max(qP + qN, 1e-300);
                  ek = Math.exp(-kl * dt); favg = kl * dt > 1e-12 ? eF[c] * (1 - ek) / (kl * dt) : eF[c];
                } else {
                  // Coulomb dispersion: the loss rate scales with the cloud's own
                  // density, so F decays hyperbolically, F/(1 + k dt), not e^(-k dt)
                  kl = Math.abs(q) * Zc * Math.abs(rq) / EPS0; col[4 * ncol] = Nde[c] * eF[c] / (A * v); col[4 * ncol + 1] = a; col[4 * ncol + 2] = q; col[4 * ncol + 3] = v; ncol++;
                  const kd = kl * dt; ek = 1 / (1 + kd); favg = kd > 1e-9 ? eF[c] * Math.log(1 + kd) / kd : eF[c];
                }
                e_dm[c] = md * dt; e_dT[c] = (Teq - T0) * (1 - ex); e_dq[c] = dq * dt; e_dF[c] = eF[c] * (ek - 1);
                // exchange-weighted survival: a drop that equilibrates (tauT) before
                // it reaches the wall has already done its exchange, so weight the
                // removal by the exchange's own time profile, not the step average
                const fw = eF[c] * exch_weight(1 / tauT, kl, dt, !corona);
                e_cond[c] = Nde[c] * fw * md * dt;
                dW -= Nde[c] * fw * md * dt; dH -= Nde[c] * fw * qc * dt;
                Cl += Nde[c] * eF[c] * m * CP_L; ClT += Nde[c] * eF[c] * m * CP_L * eT[c]; Lq += Nde[c] * eF[c] * m;
              }
            }
            // mist increments (quasi-steady drop temperature)
            const m_fav = new Float64Array(nb), m_dm = new Float64Array(nb), m_dq = new Float64Array(nb), m_k = new Float64Array(nb), m_dt = new Float64Array(nb);
            for (let b = 0; b < nb; b++) {
              const m = Math.max(mm[b], 1e-30), a = Math.cbrt(m / (RHO_L * 4 / 3 * PI)), vt = v_term(a, rho), v = Math.max(U - vt, 0.05 * U), dt = h / v;
              const Td = drop_Teq(a, mq[b], Tg, pv, vt, rho);
              let [md, qc] = drop_rates(a, mq[b], Td, Tg, pv, vt, rho);
              const mc = evap_cut(md, m, m0[b]); if (md !== 0) qc *= mc / md; md = mc;
              if (md < 0 && -md * dt > 0.5 * m) { qc *= (0.5 * m / dt) / -md; md = -0.5 * m / dt; }
              let kk = 0, dq = 0;
              if (corona) { const [qn, qP, qN] = charge_step(a, mq[b], csz, pol_c, Tg, dt); dq = (qn - mq[b]) / dt; kk = 2 * (qP + qN) * csz.E_wall * cunn(a) / (6 * PI * MU_G * a) / R; }
              else if (ncol) kk = scav(a, v, col, ncol);
              m_dm[b] = md * dt; m_dq[b] = dq * dt; m_k[b] = kk; m_dt[b] = dt;
              const Nf = Ndm[b] * (kk * dt > 1e-9 ? mF[b] * (1 - Math.exp(-kk * dt)) / (kk * dt) : mF[b]);
              m_fav[b] = Nf;
              dW -= Nf * md * dt; dH -= Nf * qc * dt;
              Cl += Nf * m * CP_L; ClT += Nf * m * CP_L * Td; Lq += Nf * m;
            }
            // falling-spray sources (per length) and the joint-equilibrium cap: th
            // scales only the heat/mass exchange, never charging or collection
            dW += S_v[k] * h / dz; dH += S_h[k] * h / dz;
            const Cg = mair * (CP_A + w * CP_V);
            let dTg = dH / Cg, dw = dW / mair, th = 1;
            // overshoot guard: an explicit gas step must not carry the gas past
            // the liquid's surface state (flux-weighted drop T and saturation
            // there); the path to the joint equilibrium is not monotone, so the
            // guard is against the driving potential's sign, not the end state
            if (Cl > 0 && !opt.nocap) {
              const Tl = ClT / Cl, wl = w_from_pv(psat(Tl));
              // only an unstable step (lands farther past it than it started) is cut back
              if (Math.abs(Tg + dTg - Tl) > Math.abs(Tg - Tl) && (Tg + dTg - Tl) * (Tg - Tl) < 0) th = Math.min(th, (Tl - Tg) / dTg);
              if (Math.abs(w + dw - wl) > Math.abs(w - wl) && (w + dw - wl) * (w - wl) < 0) th = Math.min(th, (wl - w) / dw);
              if (th < 1) nCap++;
            }
            Tg += th * dTg; w = Math.max(w + th * dw, 1e-6);
            for (let c = 0; c < NS; c++) if (sprayOn && isEnt[c] && eF[c] > 1e-9) {
              const before = eF[c] * em[c], em_old = em[c];
              em[c] = Math.max(em[c] + th * e_dm[c], 1e-30); eT[c] += th * e_dT[c]; eq[c] += e_dq[c];
              eF[c] = Math.max(eF[c] + e_dF[c], 0);
              // exact closure: what left the flux minus what condensed onto it
              const loss = Nde[c] * (before - eF[c] * em[c]) + th * e_cond[c];
              eE[c] += e_toE[c] * loss; eW[c] += (1 - e_toE[c]) * loss;
            }
            for (let b = 0; b < nb; b++) {
              const ek = Math.exp(-m_k[b] * m_dt[b]), before = mF[b] * mm[b];
              mm[b] = Math.max(mm[b] + th * m_dm[b], 1e-30); mq[b] += m_dq[b]; mF[b] *= ek;
              mD[b] += Ndm[b] * (before - mF[b] * mm[b]) + th * m_fav[b] * m_dm[b];
            }
          }
          if (zb >= (k + 0.5) * dz - 1e-12 && za <= (k + 0.5) * dz + 1e-12) {
            Tp[k] = Tg; wp[k] = w;
            let lm = 0, ls = 0;
            for (let b = 0; b < nb; b++) lm += Ndm[b] * mF[b] * mm[b];
            if (sprayOn) for (let c = 0; c < NS; c++) if (isEnt[c]) ls += Nde[c] * eF[c] * em[c];
            lqm[k] = lm; lqs[k] = ls;
            let r = 0;
            if (sprayOn && !corona) for (let c = 0; c < NS; c++) if (isEnt[c]) { const a = Math.cbrt(Math.max(em[c], 1e-30) / (RHO_L * 4 / 3 * PI)), v = Math.max(U - v_term(a, rho_in), 0.05 * U); r += Nde[c] * eF[c] / (A * v) * eq[c]; }
            rqo[k] = r;
          }
        }
      }
      // spray mass that left as liquid at the top / was lost
      let top = 0, wall = 0, elec = 0;
      for (let c = 0; c < NS; c++) if (isEnt[c]) { top += Nde[c] * eF[c] * em[c]; wall += eW[c]; elec += eE[c]; }
      let mistOut = 0, mistIn = 0, mistDep = 0;
      for (let b = 0; b < nb; b++) { mistOut += Ndm[b] * mF[b] * mm[b]; mistIn += Ndm[b] * m0[b]; mistDep += mD[b]; }
      return { Iacc, Tg, w, Tp, wp, lqm, lqs, rqe: rqo, top, wall, elec, mistOut, mistIn, mistDep, maxRq, nCap, mm, mF, mq, em, eF, eq };
    }

    // ---- outlet accounting
    const vapor_in = mair * w_in, vapor_out = mair * st.w;
    const liq_out = st.mistOut + st.top + fst.top;
    const total_in = vapor_in + st.mistIn, total_out = vapor_out + liq_out;
    const spray_lost = st.wall + st.elec + fst.wall + fst.elec + fst.bottom;
    const residual = (total_in + spray_in - total_out - st.mistDep - spray_lost) / Math.max(total_in, 1e-12);
    const vmax = max_vapor_removal(mair, T_in, w_in, spray_in, x.spray_T_C);
    let rho_launch = 0;
    for (let c = 0; c < NS; c++) if (isEnt[c]) rho_launch += Nde[c] * sp_q[c] / (A * Math.max(U - v_term(sp_a[c], rho_in), 0.05 * U));
    let maxRqf = 0; for (let k = 0; k < nz; k++) maxRqf = Math.max(maxRqf, Math.abs(rqf[k] + rqe[k]));
    const E_sc = Qtot > 0 ? Math.max(maxRqf, st.maxRq, Math.abs(rho_launch)) * R / (2 * EPS0) : 0;
    // pressure drop
    const re_t = rho_in * U * 2 * R / MU_G, fd = 0.316 * Math.pow(re_t, -0.25);
    const dp_f = fd * (L / (2 * R)) * 0.5 * rho_in * U * U, b2 = x.disc_diameter_frac ** 2;
    const nd = Math.max(1, Math.floor(Lc / x.disc_spacing_m)), cc = 0.61 + 0.39 * (1 - b2) ** 3, kd = (1 / (cc * (1 - b2)) - 1) ** 2;
    const shd = Math.min(1, x.disc_spacing_m / (4 * x.disc_diameter_frac * 2 * R));
    let drag = fst.drag;
    for (let c = 0; c < NS; c++) if (isEnt[c]) drag += Nde[c] * em0[c] * (v0s + U) + Nde[c] * em0[c] * G / Math.max(U - v_term(sp_a[c], rho_in), 0.05 * U) * (L - z_noz);
    const dp = dp_f + nd * kd * shd * 0.5 * rho_in * U * U + Math.abs(drag) / A;
    const I_c = cs.base ? st.Iacc : I_corona;
    const fan_W = dp * x.air_flow_m3_s / FAN_EFF, elec_W = V_c * I_c;
    const rec = (total_in - total_out) * 3600;
    return {
      outlet_total_water_frac: total_in > 0 ? total_out / total_in : 1,
      vapor_removed_frac: (w_in - st.w) / w_in,
      liquid_escape_frac: liq_out / Math.max(st.mistIn + spray_in, 1e-12),
      recovered_L_h: rec,
      electrical_power_W: elec_W, fan_power_W: fan_W, pressure_drop_Pa: dp,
      wall_field_kV_cm: cs.E_wall / 1e5,
      arc_margin: Math.min(arc_c, 1 - E_sc / E_BREAKDOWN),
      corona_current_mA: I_c * 1e3, corona_current_no_mist_mA: I_corona * 1e3, corona_onset_kV: cs.V_on / 1e3,
      ceiling_gap_frac: (vmax - (vapor_in - vapor_out)) / vapor_in,
      vapor_ceiling_frac: vmax / vapor_in,
      water_balance_residual: residual,
      specific_energy_Wh_L: (elec_W + fan_W) / Math.max(rec, 1e-3),
      spray_space_charge_kV_cm: E_sc / 1e5,
      mist_escape_frac: st.mistOut / Math.max(st.mistIn, 1e-12),
      mist_escape_bins: Array.from(st.mF.slice(0, nb)),
      spray_top_frac: spray_in > 0 ? (st.top + fst.top) / spray_in : 0,
      outlet_T_C: st.Tg, n_capped: st.nCap,
      vapor_in_L_h: vapor_in * 3600, mist_in_L_h: st.mistIn * 3600,
      profile: opt.profile ? { dz, Ls, L, T: Array.from(st.Tp), w: Array.from(st.wp), mist: Array.from(st.lqm), spray: Array.from(st.lqs), w_in, mist_in: st.mistIn, mair, falling: anyFall } : null,
    };
  }

  const api = { simulate, corona_rom, corona_refit, rayleigh, psat, ppf, max_vapor_removal };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.WCModel = api;
})(typeof self !== "undefined" ? self : this);
