We have run your aggregator script and it produced {figure_count} figure(s). The script's output is:
```
{aggregator_out}
```

Best figure count so far: {best_count}. Candidates with fewer figures will be rejected.

Please criticize the current script for any flaws including but not limited to:
- Are these enough plots for a final paper submission? Don't create more than {max_figures} plots.
- Have you made sure to both use key numbers and generate more detailed plots from .npy files?
- Does the figure title and legend have informative and descriptive names? These plots are the final versions, ensure there are no comments or other notes.
- Can you aggregate multiple plots into one figure if suitable?
- Do the labels have underscores? If so, replace them with spaces.
- Make sure that every plot is unique and not duplicated from the original plots.

If you believe you are done, simply say: "I am done". Otherwise, please provide an updated aggregator script in triple backticks.
