// The watercap reel: the marketing cut of watercap_script.js. Same world, shots and camera; the
// callouts say what the simulation includes (colour = stage) instead of how each piece validates.
// The status film stays the validation page's; this one is for showing the product.
import base from "./watercap_script.js";

const STAGE = { gas: "#5b9dff", spray: "#3ccfb4", mist: "#9fd8ff", corona: "#ffb547", energy: "#e8e8ec" };

// per base callout id: [stage, label, detail]
const COPY = {
  gasflow: ["gas", "k-ω SST turbulence", "The turbulent updraft is resolved from wall to axis and cross-checked against OpenFOAM."],
  dp:      ["gas", "Pressure drop", "Spray, wall and disc drag are all counted, so every design pays for the airflow it costs."],
  mist:    ["mist", "Fog transport & growth", "Every droplet size class is tracked as it grows, evaporates and settles."],
  nozzle:  ["spray", "Real nozzles", "Drop size, pressure and Cd come from measured atomizer data."],
  traj:    ["spray", "Lagrangian drops", "Thousands of drops, each with full inertia, drag and turbulent dispersion."],
  drops:   ["spray", "Drop heat & mass transfer", "Vapour condenses on every drop, with conduction inside it."],
  film:    ["spray", "Falling wall film", "Deposited spray and condensate drain to the sump, with film resistance."],
  splash:  ["mist", "Phoretic deposition", "Fog drifts to the cold wall by thermophoresis and Stefan flow."],
  flood:   ["spray", "Flooding & entrainment", "Drops the updraft carries away are followed into the corona stage."],
  onset:   ["corona", "Corona onset", "An ionization-integral onset is computed on every disc edge."],
  vi:      ["corona", "Ion drift & space charge", "A self-consistent ion field and current, with mobility tuned for humid air."],
  charge:  ["corona", "Droplet charging", "Fog droplets charge by field and diffusion charging, then drift to the wall."],
  deposit: ["corona", "Wall capture", "Charged fog lands on the grounded wall and drains away."],
  quench:  ["corona", "Mist quenching", "Dense fog chokes the corona, and the solver finds the self-consistent state."],
  coag:    ["spray", "Spray scavenging", "Falling drops sweep fog out of the air, and the captured fog follows its drop."],
  energy:  ["energy", "Full energy accounting", "Corona, pump and recooling power are reported per litre recovered."],
};

const cap = (shot, kicker, title, line, inT) => ({ shot, kicker, title, line, status: null, in: inT ?? 0.4 });

function captions(ctx) {
  return [
    cap("inlet", "Inlet", "Warm, misty exhaust rises", "Saturated exhaust enters at the bottom, carrying fog and vapour."),
    cap("nozzle", "Spray stage", "A cold spray meets it", "Chilled water is sprayed down through the rising air."),
    cap("heat", "Spray stage", "Vapour condenses on every drop", "Drops run from cold blue to warm as they pull water out of the air."),
    cap("film", "Spray stage", "The wall drains it away", "Spray and condensate that reach the wall fall to the sump as a film."),
    cap("entrain", "Spray stage", "Nothing gets lost", "Fine drops the updraft lifts are followed into the next stage."),
    cap("corona", "Corona stage", "Ions stream from every disc", "High voltage on the central rod fills the gap with ions."),
    cap("charge", "Corona stage", "Fog charges and flies to the wall", "Droplets too fine for any spray get charged and driven out of the air."),
    cap("quench", "Corona stage", "Coupled, end to end", "The fog and the field shape each other, and the solver resolves both together."),
    cap("outlet", "Outlet", "Drier air out, water back", "Recovered water, power and pressure drop for every design."),
    cap("overview", "StoneLab", "One simulation, every stage",
      "Turbulence, spray, film, fog, corona and energy are solved together, so you can explore designs in the browser.", 1.2),
  ];
}

function callouts(ctx) {
  const all = base.callouts(ctx);
  // overview: the base film pins coagulation at the very top; scavenging belongs at the spray stage's top,
  // where the base pins fog transport -- swap the two anchors
  const ovc = id => all.find(c => c.style === "compact" && c.phys === id);
  const a = ovc("coag"), b = ovc("mist");
  if (a && b) [a.at, b.at] = [b.at, a.at];
  return all.flatMap(c => {
    const k = COPY[c.phys];
    if (!k) return [];
    const [stage, label, detail] = k;
    let det = detail;
    if (c.phys === "drops" && typeof c.detail === "function") {
      const f = c.detail;   // keep the live "this drop: 300 µm, now 14.2 °C." readout
      det = (t, cx) => { const m = String(f(t, cx)).match(/^This drop: .*? °C\./); return m ? `${m[0]} ${detail}` : detail; };
    }
    return [{ ...c, label, detail: c.style === "compact" ? undefined : det, status: "inc", color: STAGE[stage] }];
  });
}

function shots(ctx) {
  return base.shots(ctx).map(s => {
    if (s.id === "open") return { ...s, chapter: "The tube",
      titleCard: { ...s.titleCard, kicker: "StoneLab", title: "Water capture, simulated",
        sub: "Warm, humid exhaust goes in at the bottom. A cold spray and then a corona pull the water out of it, and drier air leaves the top.",
        meta: "Spray · film · fog · corona · energy, all in one coupled model" } };
    if (s.id === "inlet") return { ...s, chapter: "Inlet" };
    if (s.id === "overview") return { ...s, chapter: "What's inside", params: { ...s.params, legend: 0 } };
    return s;
  });
}

export default {
  ...base,
  title: "Water Capture Reel",
  fileName: "watercap-reel",
  status: { inc: { color: STAGE.spray, label: "" } },
  legend: null,
  corner: () => ["StoneLab · water capture tube", "Air and drops at ⅕ speed"],
  shots, captions, callouts,
};
