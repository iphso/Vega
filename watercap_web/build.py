"""Inline the model, GPU kernel, corona fit and stored designs into workbench.html.
Refresh the stored designs first with scripts/export_watercap_workbench.py."""
import json
from pathlib import Path

H = Path(__file__).parent
s = (H / "workbench.src.html").read_text()
fit = json.dumps(json.load(open(H / "corona_fit.json")), separators=(",", ":"))
designs = (H / "designs.json").read_text()
# the resolved tier's worker bundle (StoneLab apps/watercap, npm run build in that worktree)
WORKER = Path.home() / "Projects/StoneLab-ehd/apps/watercap/dist/watercap-worker.js"
resworker = WORKER.read_text().replace("</script", "<\\/script")
for key, val in [("/*__WCMODEL__*/", (H / "wcmodel.js").read_text()), ("/*__WCGPU__*/", (H / "wcgpu.js").read_text()),
                 ("/*__FIT__*/null", fit), ("/*__DESIGNS__*/null", designs), ("/*__RESWORKER__*/", resworker)]:
    assert s.count(key) == 1, key
    s = s.replace(key, val)
(H / "workbench.html").write_text(s)
print(f"workbench.html {len(s) / 1e3:.0f} kB")
