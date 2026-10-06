"""Fill demo_template.html with docs/model.json and write docs/index.html."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
model = json.loads((ROOT / "docs" / "model.json").read_text())
html = (ROOT / "scripts" / "demo_template.html").read_text()
for key, val in {"__MODEL__": json.dumps(model, separators=(",", ":")), "__SCRAPED__": model["scraped"],
                 "__MINCELL__": str(model["min_per_cell"]), "__NTRAIN__": f"{model['n_train']:,}",
                 "__MAE__": f"{model['cv_mae']:.2f}", "__MAESD__": f"{model['cv_mae_sd']:.2f}", "__MDAPE__": f"{model['mdape']*100:.0f}"}.items():
    html = html.replace(key, val)
assert "__" not in html.replace("__proto__", ""), "unfilled placeholder"
(ROOT / "docs" / "index.html").write_text(html)
print(f"docs/index.html {len(html)/1e6:.2f} MB")
