from run_ablation import summarize_all_on_disk
from pathlib import Path
import json

OUT_DIR = Path("/work/output")
summary = summarize_all_on_disk()
(OUT_DIR / "ablation_summary.json").write_text(json.dumps(summary, indent=2))
print(f"rebuilt ablation_summary.json with {len(summary)} configs")
