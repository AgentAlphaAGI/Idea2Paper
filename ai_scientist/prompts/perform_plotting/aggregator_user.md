
We have three JSON summaries of scientific experiments: baseline, research, ablation.
They may contain lists of figure descriptions, code to generate the figures, and paths to the .npy files containing the numerical results.
Our goal is to produce final, publishable figures.

--- RESEARCH IDEA ---
```
{idea_text}
```

IMPORTANT:
- The aggregator script must load existing .npy experiment data by reading "plot_agg_manifest.json" in the same folder.
- .npy files MUST come ONLY from manifest["npy_files"] (absolute paths). Do NOT construct runs/experiments/... paths.
- Do NOT infer .npy paths from image paths (plot_paths/plot_path). Ignore those fields if present.
- It should call os.makedirs("figures", exist_ok=True) before saving any plots.
- Aim for a balance of empirical results, ablations, and diverse, informative visuals in 'figures/' that comprehensively showcase the finalized research outcomes.
- If you need .npy paths, read them from the manifest (not from summaries).

Minimal loading template (copy as-is):
```python
from pathlib import Path
import json
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
manifest = json.loads((BASE_DIR / "plot_agg_manifest.json").read_text(encoding="utf-8"))
npy_files = manifest["npy_files"]
# Example: data = np.load(npy_files[0], allow_pickle=True).item()
```

Your generated Python script must:
1) Load or refer to relevant data and .npy files from the manifest only (absolute paths).
2) Synthesize or directly create final, scientifically meaningful plots for a final research paper (comprehensive and complete), referencing the original code if needed to see how the data was generated.
3) Carefully combine or replicate relevant existing plotting code to produce these final aggregated plots in 'figures/' only, since only those are used in the final paper.
4) Do not hallucinate data. Data must either be loaded from .npy files or copied from the JSON summaries.
5) The aggregator script must be fully self-contained, and place the final plots in 'figures/'.
6) This aggregator script should produce a comprehensive and final set of scientific plots for the final paper, reflecting all major findings from the experiment data.
7) Make sure that every plot is unique and not duplicated from the original plots. Delete any duplicate plots if necessary.
8) Each figure can have up to 3 subplots using fig, ax = plt.subplots(1, 3).
9) Use a font size larger than the default for plot labels and titles to ensure they are readable in the final PDF paper.


Below are the summaries in JSON:

{combined_summaries_str}

Respond with a Python script in triple backticks.
