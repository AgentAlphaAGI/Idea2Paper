The abstract of the paper is:

{abstract}

You will be given an image via the vision API. As a careful scientist reviewer, your task is to:
  1. Examine the provided image closely.
  2. Describe in detail what the image shows in a scientific manner.
  3. Critically analyze whether the image content aligns with the given caption:

{caption}

  4. We also have references in the main text that mention the figure:

{main_text_figrefs}

You should:
  - Examine the figure in detail: conclude elements in figures (e.g. name of axis) and describe what information is shown (e.g. the line of loss decreases monotonically but plateaus after X epochs)
  - Suggest any potential improvements or issues in the figure itself (e.g., missing legend, unclear labeling, no meaningful conclusion, mismatch with what the caption claims).
  - Critique the caption: does it accurately describe the figure? Is it too long/short? Does it include a concise takeaway?
  - Review how well the main text references (figrefs) explain the figure: are they missing? Do they adequately describe the figure's content, context, or purpose?

Finally, respond in the following format:

THOUGHT:
<THOUGHT>

REVIEW JSON:
```json
<JSON>
```
In <JSON>, provide the review in JSON format with the following fields in the order:
- "Img_description": "<Describe the figure's contents here>"
- "Img_review": "<Your analysis of the figure itself, including any suggestions for improvement>"
- "Caption_review": "<Your assessment of how well the caption matches the figure and any suggestions>"
- "Figrefs_review": "<Your thoughts on whether the main text references adequately describe or integrate the figure>"

In <THOUGHT>, first, thoroughly reason through your observations, analysis of alignment, and any suggested improvements. It is okay to be very long.
Then provide your final structured output in <JSON>.
Make sure the JSON is valid and properly formatted, as it will be parsed automatically.
