import argparse
import json
import os
import os.path as osp
import re
import shutil
import subprocess
import traceback
import unicodedata
import uuid
import tempfile

from ai_scientist.llm import (
    get_response_from_llm,
    extract_json_between_markers,
    create_client,
    AVAILABLE_LLMS,
)

from ai_scientist.utils.token_tracker import track_token_usage

from ai_scientist.tools.semantic_scholar import search_for_papers

from ai_scientist.perform_vlm_review import (
    generate_vlm_img_review,
    perform_imgs_cap_ref_review,
    perform_imgs_cap_ref_review_selection,
    detect_duplicate_figures,
)
from ai_scientist.vlm import create_client as create_vlm_client
from ai_scientist.prompts.loader import load_prompt, render_prompt


def remove_accents_and_clean(s):
    # Normalize to separate accents
    nfkd_form = unicodedata.normalize("NFKD", s)
    # Remove non-ASCII characters
    ascii_str = nfkd_form.encode("ASCII", "ignore").decode("ascii")
    # Remove anything but letters, digits, underscores, colons, dashes, @, {, }, and commas
    ascii_str = re.sub(r"[^a-zA-Z0-9:_@\{\},-]+", "", ascii_str)
    # Convert to lowercase
    ascii_str = ascii_str.lower()
    return ascii_str


def compile_latex(cwd, pdf_file, timeout=30):
    print("GENERATING LATEX")

    commands = [
        ["pdflatex", "-interaction=nonstopmode", "template.tex"],
        ["bibtex", "template"],
        ["pdflatex", "-interaction=nonstopmode", "template.tex"],
        ["pdflatex", "-interaction=nonstopmode", "template.tex"],
    ]

    for command in commands:
        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
            )
            print("Standard Output:\n", result.stdout)
            print("Standard Error:\n", result.stderr)
        except subprocess.TimeoutExpired:
            print(
                f"EXCEPTION in compile_latex: LaTeX timed out after {timeout} seconds."
            )
            print(traceback.format_exc())
        except subprocess.CalledProcessError:
            print(
                f"EXCEPTION in compile_latex: Error running command {' '.join(command)}"
            )
            print(traceback.format_exc())

    print("FINISHED GENERATING LATEX")

    try:
        shutil.move(osp.join(cwd, "template.pdf"), pdf_file)
    except FileNotFoundError:
        print("Failed to rename PDF.")
        print("EXCEPTION in compile_latex while moving PDF:")
        print(traceback.format_exc())


def is_header_or_footer(line):
    """
    Returns True if the line is likely a header or footer.
    Filters out:
      - Lines that are too short (< 4 characters after stripping).
      - Lines that are only digits.
      - Lines starting with known phrases (e.g., "Under review").
      - Lines that consist solely of capital letters and spaces.
    """
    line_stripped = line.strip()
    if len(line_stripped) < 1:
        return True

    header_footer_patterns = [
        r"^\d+$",  # Only digits (e.g., page numbers like "000", "001", etc.)
        r"^Under review",  # Lines starting with "Under review"
    ]
    for pattern in header_footer_patterns:
        if re.match(pattern, line_stripped):
            return True
    return False


def clean_lines(content):
    """
    Given raw text content, split it into lines and remove lines that are
    likely headers/footers or otherwise not part of the main content.
    """
    lines = content.splitlines()
    # Keep only lines that are not detected as headers/footers.
    return [line for line in lines if not is_header_or_footer(line)]


def detect_references_position_clean(pdf_file):
    """
    Locate the first occurrence of the word "References" (or variations like
    "R EFERENCES") within the cleaned content extracted from the PDF.
    Uses pdftotext with layout preservation and cleans the extracted text.

    Returns a tuple (ref_page, ref_line) if found (with ref_line counting only
    the cleaned lines), otherwise None.
    """
    if not osp.exists(pdf_file):
        return None

    # Compile a regex pattern to match "REFERENCES" even if there are extra spaces
    # between letters (and do a case-insensitive match).
    pattern = re.compile(r"\bR\s*E\s*F\s*E\s*R\s*E\s*N\s*C\s*E\s*S\b", re.IGNORECASE)

    # Loop through pages (limit to 50 pages by default)
    for page in range(1, 51):
        temp_dir = tempfile.mkdtemp()
        page_txt = osp.join(temp_dir, f"page_{page}.txt")
        try:
            subprocess.run(
                [
                    "pdftotext",
                    "-layout",
                    "-f",
                    str(page),
                    "-l",
                    str(page),
                    "-q",
                    pdf_file,
                    page_txt,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if not osp.exists(page_txt):
                shutil.rmtree(temp_dir)
                break
            try:
                with open(page_txt, "r", encoding="utf-8", errors="ignore") as fp:
                    content = fp.read()
            except Exception as e:
                print(f"Error reading page {page}: {e}")
                print(traceback.format_exc())
                shutil.rmtree(temp_dir)
                continue
            finally:
                shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Error running pdftotext for page {page}: {e}")
            print(traceback.format_exc())
            shutil.rmtree(temp_dir)
            continue

        # Clean the lines before searching for "References"
        cleaned = clean_lines(content)
        for idx, line in enumerate(cleaned):
            if pattern.search(line):
                # Found "References" on this page at cleaned line number idx+1
                return (page, idx + 1)
    return None


def extract_page_line_counts(pdf_file, first_page, last_page):
    """
    Extract the number of cleaned text lines for each page from first_page to last_page.
    This uses pdftotext with layout preservation and the clean_lines helper.
    Returns a dictionary {page_number: number_of_cleaned_lines}.
    Pages for which extraction fails are omitted.
    """
    page_lines = {}
    for page in range(first_page, last_page + 1):
        temp_dir = tempfile.mkdtemp()
        page_txt = osp.join(temp_dir, f"page_{page}.txt")
        try:
            subprocess.run(
                [
                    "pdftotext",
                    "-layout",
                    "-f",
                    str(page),
                    "-l",
                    str(page),
                    "-q",
                    pdf_file,
                    page_txt,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if not osp.exists(page_txt):
                shutil.rmtree(temp_dir)
                break
            try:
                with open(page_txt, "r", encoding="utf-8", errors="ignore") as fp:
                    content = fp.read()
            except Exception as e:
                print(f"Error reading page {page}: {e}")
                print(traceback.format_exc())
                shutil.rmtree(temp_dir)
                continue
            finally:
                shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Error running pdftotext for page {page}: {e}")
            print(traceback.format_exc())
            shutil.rmtree(temp_dir)
            continue
        # Clean the extracted text and count the number of remaining lines.
        cleaned = clean_lines(content)
        page_lines[page] = len(cleaned)
    return page_lines


def check_page_limit(pdf_file, page_limit=4, timeout=30):
    """
    Compile the LaTeX project in a temporary folder, then determine where the
    "References" section begins using cleaned text extraction. Next, count the
    number of cleaned text lines used before the word "References" and compare that
    to the total number of cleaned lines available in the allowed number of pages (page_limit).

    Returns a dictionary with:
      - 'ref_page': page number where "References" was found (or None)
      - 'ref_line': cleaned line number within that page (or None)
      - 'used_lines': number of cleaned lines used for main content (before "References")
      - 'allowed_lines': total number of cleaned text lines available in pages 1..page_limit
      - 'excess': if used_lines > allowed_lines (number of lines over the limit),
      - 'available': if used_lines < allowed_lines (number of lines still available)

    If compilation or extraction fails, returns None.
    """
    try:
        # Ensure the PDF was produced
        if not osp.exists(pdf_file):
            return None

        # Locate the first occurrence of "References" using the cleaned extraction
        ref_pos = detect_references_position_clean(pdf_file)
        if ref_pos is None:
            # If "References" isn't found, assume no reference section exists.
            return None
        ref_page, ref_line = ref_pos

        # Determine up to which page we need to extract cleaned line counts:
        max_page_to_extract = max(page_limit, ref_page)
        page_line_counts = extract_page_line_counts(pdf_file, 1, max_page_to_extract)
        if not page_line_counts:
            return None

        # Compute total cleaned lines available in the allowed pages (pages 1 to page_limit)
        allowed_lines = sum(
            page_line_counts.get(page, 0) for page in range(1, page_limit + 1)
        )

        # Compute cleaned lines used before "References":
        used_lines = 0
        # Sum full pages before the reference page
        for page in range(1, ref_page):
            used_lines += page_line_counts.get(page, 0)
        # Add lines from the reference page up to (but not including) the line where "References" appears
        used_lines += ref_line - 1

        result = {
            "ref_page": ref_page,
            "ref_line": ref_line,
            "used_lines": used_lines,
            "allowed_lines": allowed_lines,
        }
        if used_lines > allowed_lines:
            result["excess"] = used_lines - allowed_lines
        else:
            result["available"] = allowed_lines - used_lines
        return result

    except Exception as e:
        print(f"Error checking page limit: {e}")
        print(traceback.format_exc())
        return None


def get_reflection_page_info(reflection_pdf, page_limit):
    info = check_page_limit(reflection_pdf, page_limit)
    if info is not None:
        if "excess" in info:
            reflection_page_info = (
                f"\nCurrently, 'References' begins on page {info['ref_page']}, approximately on line {info['ref_line']}. "
                f"The main text (before the references) uses {info['used_lines']} lines, which exceeds the allowed {info['allowed_lines']} lines for a {page_limit}-page limit by {info['excess']} lines. "
                f"DO NOT USE MORE THAN {page_limit} PAGES FOR THE MAIN TEXT. Please reduce the text or resize the plot to meet the page limit. "
                f"Consider grouping plots together to make the paper more concise. "
                f"Papers often look more professional if the main text is just under {page_limit} pages in length.\n"
            )
        elif "available" in info:
            reflection_page_info = (
                f"\nCurrently, 'References' begins on page {info['ref_page']}, approximately on line {info['ref_line']}. "
                f"The main text (before the references) uses {info['used_lines']} lines, leaving {info['available']} lines available out of the allowed {info['allowed_lines']} lines (which corresponds to {page_limit} pages). "
                f"DO NOT USE MORE THAN {page_limit} PAGES FOR THE MAIN TEXT. You can add up to {info['available']} lines if needed, "
                f"but papers often look more professional if the main text is just under {page_limit} pages in length.\n"
            )
        else:
            # Fallback in case the info dictionary doesn't contain 'excess' or 'available'
            reflection_page_info = (
                f"\nCurrently, 'References' begins on page {info.get('ref_page', '?')}, approximately on line {info.get('ref_line', '?')}. "
                f"The page limit is {page_limit} pages for the main text before the references. "
                f"DO NOT USE MORE THAN {page_limit} PAGES FOR THE MAIN TEXT. Adjust your content accordingly.\n"
            )
    else:
        reflection_page_info = (
            "\nCould not detect 'References' page (compilation or detection failed).\n"
        )

    return reflection_page_info


def get_citation_addition(
    client, model, context, current_round, total_rounds, idea_text
):
    report, citations = context
    msg_history = []
    citation_system_msg_template = load_prompt(
        "perform_icbinb_writeup.citation_system_msg_template"
    )

    citation_first_prompt_template = load_prompt(
        "perform_icbinb_writeup.citation_first_prompt_template"
    )

    citation_second_prompt_template = load_prompt(
        "perform_icbinb_writeup.citation_second_prompt_template"
    )

    try:
        text, msg_history = get_response_from_llm(
            prompt=render_prompt(
                "perform_icbinb_writeup.citation_first_prompt_template",
                current_round=current_round + 1,
                total_rounds=total_rounds,
                Idea=idea_text,
                report=report,
                citations=citations,
            ),
            client=client,
            model=model,
            system_message=render_prompt(
                "perform_icbinb_writeup.citation_system_msg_template",
                total_rounds=total_rounds,
            ),
            msg_history=msg_history,
            print_debug=False,
        )
        if "No more citations needed" in text:
            print("No more citations needed.")
            return None, True

        json_output = extract_json_between_markers(text)
        assert json_output is not None, "Failed to extract JSON from LLM output"
        query = json_output["Query"]
        papers = search_for_papers(query, result_limit=5)
    except Exception:
        print("EXCEPTION in get_citation_addition (initial search):")
        print(traceback.format_exc())
        return None, False

    if papers is None:
        print("No papers found.")
        return None, False

    paper_strings = []
    for i, paper in enumerate(papers):
        paper_strings.append(
            "{i}: {title}. {authors}. {venue}, {year}.\nAbstract: {abstract}".format(
                i=i,
                title=paper["title"],
                authors=paper["authors"],
                venue=paper["venue"],
                year=paper["year"],
                abstract=paper["abstract"],
            )
        )
    papers_str = "\n\n".join(paper_strings)

    try:
        text, msg_history = get_response_from_llm(
            prompt=render_prompt(
                "perform_icbinb_writeup.citation_second_prompt_template",
                papers=papers_str,
                current_round=current_round + 1,
                total_rounds=total_rounds,
            ),
            client=client,
            model=model,
            system_message=render_prompt(
                "perform_icbinb_writeup.citation_system_msg_template",
                total_rounds=total_rounds,
            ),
            msg_history=msg_history,
            print_debug=False,
        )
        if "Do not add any" in text:
            print("Do not add any.")
            return None, False

        json_output = extract_json_between_markers(text)
        assert json_output is not None, "Failed to extract JSON from LLM output"
        desc = json_output["Description"]
        selected_papers = str(json_output["Selected"])

        if selected_papers != "[]":
            selected_indices = []
            for x in selected_papers.strip("[]").split(","):
                x_str = x.strip().strip('"').strip("'")
                if x_str:
                    selected_indices.append(int(x_str))
            assert all(
                [0 <= i < len(papers) for i in selected_indices]
            ), "Invalid paper index"
            bibtexs = [papers[i]["citationStyles"]["bibtex"] for i in selected_indices]

            cleaned_bibtexs = []
            for bibtex in bibtexs:
                newline_index = bibtex.find("\n")
                cite_key_line = bibtex[:newline_index]
                cite_key_line = remove_accents_and_clean(cite_key_line)
                cleaned_bibtexs.append(cite_key_line + bibtex[newline_index:])
            bibtexs = cleaned_bibtexs

            bibtex_string = "\n".join(bibtexs)
        else:
            return None, False

    except Exception:
        print("EXCEPTION in get_citation_addition (selecting papers):")
        print(traceback.format_exc())
        return None, False

    references_format = """% {description}
{bibtex}"""

    references_prompt = references_format.format(bibtex=bibtex_string, description=desc)
    return references_prompt, False


writeup_system_message_template = load_prompt(
    "perform_icbinb_writeup.writeup_system_message_template"
)

writeup_prompt = load_prompt("perform_icbinb_writeup.writeup_prompt")


def load_idea_text(base_folder):
    """
    Load the idea text from the base folder.
    """
    idea_text = ""
    research_idea_path = osp.join(base_folder, "research_idea.md")
    if osp.exists(research_idea_path):
        with open(research_idea_path, "r") as f_idea:
            idea_text = f_idea.read()
    else:
        idea_md_path = osp.join(base_folder, "idea.md")
        if osp.exists(idea_md_path):
            with open(idea_md_path, "r") as f_idea:
                idea_text = f_idea.read()
    return idea_text


def load_exp_summaries(base_folder):
    """
    Load the experiment summaries from the base folder.
    """
    summary_files = [
        ("logs/0-run/baseline_summary.json", "BASELINE_SUMMARY"),
        ("logs/0-run/research_summary.json", "RESEARCH_SUMMARY"),
        ("logs/0-run/ablation_summary.json", "ABLATION_SUMMARY"),
    ]
    loaded_summaries = {}
    for fname, key in summary_files:
        path = osp.join(base_folder, fname)
        if osp.exists(path):
            try:
                with open(path, "r") as f:
                    loaded_summaries[key] = json.load(f)
            except json.JSONDecodeError:
                print(
                    f"Warning: {fname} is not valid JSON. Using empty data for {key}."
                )
                loaded_summaries[key] = {}
        else:
            loaded_summaries[key] = {}
    return loaded_summaries


def filter_experiment_summaries(exp_summaries, step_name):
    def strip_plot_path(plot_analyses):
        if isinstance(plot_analyses, list):
            cleaned = []
            for entry in plot_analyses:
                if isinstance(entry, dict):
                    cleaned.append({k: v for k, v in entry.items() if k != "plot_path"})
                else:
                    cleaned.append(entry)
            return cleaned
        return plot_analyses

    if step_name == "citation_gathering":
        node_keys_to_keep = {
            "overall_plan",
            "analysis",
            "metric",
            "vlm_feedback_summary",
        }
    elif step_name == "writeup":
        node_keys_to_keep = {
            "overall_plan",
            "analysis",
            "metric",
            "code",
            "plot_analyses",
            "vlm_feedback_summary",
        }
    elif step_name == "plot_aggregation":
        node_keys_to_keep = {
            "overall_plan",
            "analysis",
            "plot_plan",
            "plot_code",
            "plot_analyses",
            "vlm_feedback_summary",
            "exp_results_npy_files",
        }
    else:
        raise ValueError(f"Invalid step name: {step_name}")

    filtered_summaries = {}
    for stage_name in exp_summaries.keys():
        if stage_name in {"BASELINE_SUMMARY", "RESEARCH_SUMMARY"}:
            filtered_summaries[stage_name] = {}
            for key in exp_summaries[stage_name].keys():
                if key in {"best node"}:
                    filtered_summaries[stage_name][key] = {}
                    for node_key in exp_summaries[stage_name][key].keys():
                        if node_key in node_keys_to_keep:
                            value = exp_summaries[stage_name][key][node_key]
                            if (
                                step_name == "plot_aggregation"
                                and node_key == "plot_analyses"
                            ):
                                value = strip_plot_path(value)
                            filtered_summaries[stage_name][key][node_key] = value
        elif stage_name == "ABLATION_SUMMARY" and step_name == "plot_aggregation":
            filtered_summaries[stage_name] = {}
            for ablation_summary in exp_summaries[stage_name]:
                filtered_summaries[stage_name][ablation_summary["ablation_name"]] = {}
                for node_key in ablation_summary.keys():
                    if node_key in node_keys_to_keep:
                        value = ablation_summary[node_key]
                        if node_key == "plot_analyses":
                            value = strip_plot_path(value)
                        filtered_summaries[stage_name][
                            ablation_summary["ablation_name"]
                        ][node_key] = value
    return filtered_summaries


def gather_citations(base_folder, num_cite_rounds=20, small_model="gpt-4o-2024-05-13"):
    """
    Gather citations for a paper, with ability to resume from previous progress.

    Args:
        base_folder: Path to project folder
        num_cite_rounds: Maximum number of citation gathering rounds
        small_model: Model to use for citation collection
        resume: Whether to try to resume from previous progress

    Returns:
        str: The gathered citations text, or None if failed
    """

    # Paths for storing progress
    citations_cache_path = osp.join(base_folder, "cached_citations.bib")
    progress_path = osp.join(base_folder, "citations_progress.json")

    # Initialize or load progress
    current_round = 0
    citations_text = ""

    if osp.exists(citations_cache_path) and osp.exists(progress_path):
        try:
            with open(citations_cache_path, "r") as f:
                citations_text = f.read()
            with open(progress_path, "r") as f:
                progress = json.load(f)
                current_round = progress.get("completed_rounds", 0)
            print(f"Resuming citation gathering from round {current_round}")
        except Exception as e:
            print(f"Error loading cached citations: {e}")
            print("Starting fresh")
            current_round = 0
            citations_text = ""

    try:
        # Load idea text and summaries
        idea_text = load_idea_text(base_folder)
        exp_summaries = load_exp_summaries(base_folder)
        filtered_summaries = filter_experiment_summaries(
            exp_summaries, step_name="citation_gathering"
        )
        filtered_summaries_str = json.dumps(filtered_summaries, indent=2)

        # Run small model for citation additions
        client, client_model = create_client(small_model)

        for round_idx in range(current_round, num_cite_rounds):
            try:
                context_for_citation = (filtered_summaries_str, citations_text)
                addition, done = get_citation_addition(
                    client,
                    client_model,
                    context_for_citation,
                    round_idx,
                    num_cite_rounds,
                    idea_text,
                )

                if done:
                    # Save final state before exiting
                    with open(citations_cache_path, "w") as f:
                        f.write(citations_text)
                    with open(progress_path, "w") as f:
                        json.dump(
                            {"completed_rounds": round_idx + 1, "status": "completed"},
                            f,
                        )
                    break

                if addition is not None:
                    # Simple check to avoid duplicating the same title
                    title_match = re.search(r" title = {(.*?)}", addition)
                    if title_match:
                        new_title = title_match.group(1).lower()
                        existing_titles = re.findall(
                            r" title = {(.*?)}", citations_text
                        )
                        existing_titles = [t.lower() for t in existing_titles]
                        if new_title not in existing_titles:
                            citations_text += "\n" + addition
                            # Save progress after each successful addition
                            with open(citations_cache_path, "w") as f:
                                f.write(citations_text)
                            with open(progress_path, "w") as f:
                                json.dump(
                                    {
                                        "completed_rounds": round_idx + 1,
                                        "status": "in_progress",
                                    },
                                    f,
                                )

            except Exception as e:
                print(f"Error in citation round {round_idx}: {e}")
                print(traceback.format_exc())
                # Save progress even if there's an error
                with open(citations_cache_path, "w") as f:
                    f.write(citations_text)
                with open(progress_path, "w") as f:
                    json.dump({"completed_rounds": round_idx, "status": "error"}, f)
                continue

        return citations_text if citations_text else None

    except Exception:
        print("EXCEPTION in gather_citations:")
        print(traceback.format_exc())
        return citations_text if citations_text else None


def perform_writeup(
    base_folder,
    citations_text=None,
    no_writing=False,
    num_cite_rounds=20,
    small_model="gpt-4o-2024-05-13",
    big_model="o1-2024-12-17",
    n_writeup_reflections=3,
    page_limit=4,
):
    pdf_file = osp.join(base_folder, f"{osp.basename(base_folder)}.pdf")
    latex_folder = osp.join(base_folder, "latex")

    # Cleanup any previous latex folder and pdf
    if osp.exists(latex_folder):
        shutil.rmtree(latex_folder)
    if osp.exists(pdf_file):
        os.remove(pdf_file)

    # Remove any previous reflection PDFs
    for old_pdf in os.listdir(base_folder):
        if old_pdf.endswith(".pdf") and "reflection" in old_pdf:
            os.remove(osp.join(base_folder, old_pdf))

    try:
        idea_text = load_idea_text(base_folder)
        exp_summaries = load_exp_summaries(base_folder)
        filtered_summaries_for_writeup = filter_experiment_summaries(
            exp_summaries, step_name="writeup"
        )
        # Convert them to one big JSON string for context
        combined_summaries_str = json.dumps(filtered_summaries_for_writeup, indent=2)

        # Prepare a new fresh latex folder
        if not osp.exists(osp.join(latex_folder, "template.tex")):
            shutil.copytree(
                "ai_scientist/blank_icbinb_latex", latex_folder, dirs_exist_ok=True
            )

        writeup_file = osp.join(latex_folder, "template.tex")
        with open(writeup_file, "r") as f:
            writeup_text = f.read()

        # Gather plot filenames from figures/ folder
        figures_dir = osp.join(base_folder, "figures")
        plot_names = []
        if osp.exists(figures_dir):
            for fplot in os.listdir(figures_dir):
                if fplot.lower().endswith(".png"):
                    plot_names.append(fplot)

        # Load aggregator script to include in the prompt
        aggregator_path = osp.join(base_folder, "auto_plot_aggregator.py")
        aggregator_code = ""
        if osp.exists(aggregator_path):
            with open(aggregator_path, "r") as fa:
                aggregator_code = fa.read()
        else:
            aggregator_code = "No aggregator script found."

        if no_writing:
            compile_latex(latex_folder, pdf_file)
            return osp.exists(pdf_file)

        # If no citations provided, try to load from cache first
        if citations_text is None:
            citations_cache_path = osp.join(base_folder, "cached_citations.bib")
            if osp.exists(citations_cache_path):
                try:
                    with open(citations_cache_path, "r") as f:
                        citations_text = f.read()
                    print("Loaded citations from cache")
                except Exception as e:
                    print(f"Error loading cached citations: {e}")
                    citations_text = None

            # If still no citations, gather them
            if not citations_text:
                citations_text = gather_citations(
                    base_folder, num_cite_rounds, small_model
                )
                if citations_text is None:
                    print("Warning: Citation gathering failed")
                    citations_text = ""

        # Insert citations into template.tex
        if citations_text:
            with open(writeup_file, "r") as f:
                content = f.read()
            pattern_end = r"\end{filecontents}"
            content = content.replace(pattern_end, f"\n{citations_text}{pattern_end}")
            with open(writeup_file, "w") as f:
                f.write(content)

        # Generate VLM-based descriptions
        try:
            vlm_client, vlm_model = create_vlm_client(small_model)
            desc_map = {}
            for pf in plot_names:
                ppath = osp.join(figures_dir, pf)
                if not osp.exists(ppath):
                    continue
                img_dict = {
                    "images": [ppath],
                    "caption": "No direct caption",
                }
                review_data = generate_vlm_img_review(img_dict, vlm_model, vlm_client)
                if review_data:
                    desc_map[pf] = review_data.get(
                        "Img_description", "No description found"
                    )
                else:
                    desc_map[pf] = "No description found"

            plot_descriptions_list = []
            for fname in plot_names:
                desc_text = desc_map.get(fname, "No description found")
                plot_descriptions_list.append(f"{fname}: {desc_text}")
            plot_descriptions_str = "\n".join(plot_descriptions_list)
        except Exception:
            print("EXCEPTION in VLM figure description generation:")
            print(traceback.format_exc())
            plot_descriptions_str = "No descriptions available."

        big_model_system_message = render_prompt(
            "perform_icbinb_writeup.writeup_system_message_template",
            page_limit=page_limit,
        )
        big_client, big_client_model = create_client(big_model)
        with open(writeup_file, "r") as f:
            writeup_text = f.read()

        combined_prompt = render_prompt(
            "perform_icbinb_writeup.writeup_prompt",
            idea_text=idea_text,
            summaries=combined_summaries_str,
            aggregator_code=aggregator_code,
            plot_list=", ".join(plot_names),
            latex_writeup=writeup_text,
            plot_descriptions=plot_descriptions_str,
        )

        response, msg_history = get_response_from_llm(
            prompt=combined_prompt,
            client=big_client,
            model=big_client_model,
            system_message=big_model_system_message,
            print_debug=False,
        )

        latex_code_match = re.search(r"```latex(.*?)```", response, re.DOTALL)
        if not latex_code_match:
            return False
        updated_latex_code = latex_code_match.group(1).strip()
        with open(writeup_file, "w") as f:
            f.write(updated_latex_code)

        # Multiple reflection loops on the final LaTeX
        for i in range(n_writeup_reflections):
            with open(writeup_file, "r") as f:
                current_latex = f.read()

            # Check for unused or invalid figure references
            referenced_figs_temp = re.findall(
                r"\\includegraphics(?:\[[^\]]*\])?{([^}]+)}", current_latex
            )
            used_figs = set(os.path.basename(fig) for fig in referenced_figs_temp)
            all_figs = set(plot_names)
            unused_figs = all_figs - used_figs
            invalid_figs = used_figs - all_figs

            # Save PDF with reflection trial number
            reflection_pdf = osp.join(
                base_folder, f"{osp.basename(base_folder)}_reflection{i+1}.pdf"
            )
            # Compile current version before reflection
            print(f"[green]Compiling PDF for reflection {i+1}...[/green]")
            compile_latex(latex_folder, reflection_pdf)

            review_img_cap_ref = perform_imgs_cap_ref_review(
                vlm_client, vlm_model, reflection_pdf
            )

            # Detect duplicate figures between main text and appendix
            analysis_duplicate_figs = detect_duplicate_figures(
                vlm_client, vlm_model, reflection_pdf
            )
            print(analysis_duplicate_figs)

            # Get reflection_page_info
            reflection_page_info = get_reflection_page_info(reflection_pdf, page_limit)

            check_output = os.popen(  # TODO: should prob use subprocess instead
                f"chktex {writeup_file} -q -n2 -n24 -n13 -n1"
            ).read()

            reflection_prompt = render_prompt(
                "perform_icbinb_writeup.reflection_prompt",
                unused_figs=sorted(unused_figs),
                invalid_figs=sorted(invalid_figs),
                reflection_page_info=reflection_page_info,
                check_output=check_output,
                review_img_cap_ref=review_img_cap_ref,
                analysis_duplicate_figs=analysis_duplicate_figs,
            )

            reflection_response, msg_history = get_response_from_llm(
                prompt=reflection_prompt,
                client=big_client,
                model=big_client_model,
                system_message=big_model_system_message,
                msg_history=msg_history[-1:],
                print_debug=False,
            )

            # 2nd run:
            reflection_code_match = re.search(
                r"```latex(.*?)```", reflection_response, re.DOTALL
            )
            if reflection_code_match:
                reflected_latex_code = reflection_code_match.group(1).strip()
                if reflected_latex_code != current_latex:
                    final_text = reflected_latex_code
                    cleanup_map = {
                        "</end": r"\\end",
                        "</begin": r"\\begin",
                        "’": "'",
                    }
                    for bad_str, repl_str in cleanup_map.items():
                        final_text = final_text.replace(bad_str, repl_str)
                    final_text = re.sub(r"(\d+(?:\.\d+)?)%", r"\1\\%", final_text)

                    with open(writeup_file, "w") as fo:
                        fo.write(final_text)

                    compile_latex(latex_folder, reflection_pdf)
                else:
                    print(f"No changes in reflection step {i+1}.")
                    break
            else:
                print(f"No valid LaTeX code block found in reflection step {i+1}.")
                break
            # Get new reflection_page_info
            reflection_page_info = get_reflection_page_info(reflection_pdf, page_limit)
            review_img_selection = perform_imgs_cap_ref_review_selection(
                vlm_client, vlm_model, reflection_pdf, reflection_page_info
            )
            img_reflection_prompt = render_prompt(
                "perform_icbinb_writeup.img_reflection_prompt",
                used_figs=sorted(used_figs),
                unused_figs=sorted(unused_figs),
                reflection_page_info=reflection_page_info,
                review_img_selection=review_img_selection,
            )
            reflection_response, msg_history = get_response_from_llm(
                prompt=img_reflection_prompt,
                client=big_client,
                model=big_client_model,
                system_message=big_model_system_message,
                msg_history=msg_history[-1:],
                print_debug=False,
            )

            if "I am done" in reflection_response:
                print(
                    "LLM indicated it is done with reflections. Exiting reflection loop."
                )
                break

            reflection_code_match = re.search(
                r"```latex(.*?)```", reflection_response, re.DOTALL
            )
            if reflection_code_match:
                reflected_latex_code = reflection_code_match.group(1).strip()
                if reflected_latex_code != current_latex:
                    final_text = reflected_latex_code
                    cleanup_map = {
                        "</end": r"\\end",
                        "</begin": r"\\begin",
                        "’": "'",
                    }
                    for bad_str, repl_str in cleanup_map.items():
                        final_text = final_text.replace(bad_str, repl_str)
                    final_text = re.sub(r"(\d+(?:\.\d+)?)%", r"\1\\%", final_text)

                    with open(writeup_file, "w") as fo:
                        fo.write(final_text)

                    compile_latex(latex_folder, reflection_pdf)
                else:
                    print(f"No changes in reflection step {i+1}.")
                    break
            else:
                print(f"No valid LaTeX code block found in reflection step {i+1}.")
                break

        # Final reflection on page limit
        # Save PDF with reflection

        # Get new reflection_page_info
        reflection_page_info = get_reflection_page_info(reflection_pdf, page_limit)

        final_reflection_prompt = load_prompt(
            "perform_icbinb_writeup.final_reflection_prompt"
        )
        reflection_response, msg_history = get_response_from_llm(
            prompt=final_reflection_prompt,
            client=big_client,
            model=big_client_model,
            system_message=big_model_system_message,
            msg_history=msg_history[-1:],
            print_debug=False,
        )

        reflection_pdf = osp.join(
            base_folder, f"{osp.basename(base_folder)}_reflection_final_page_limit.pdf"
        )
        # Compile current version before reflection
        print(f"[green]Compiling PDF for reflection final page limit...[/green]")

        print(f"reflection step {i+1}")

        reflection_code_match = re.search(
            r"```latex(.*?)```", reflection_response, re.DOTALL
        )
        if reflection_code_match:
            reflected_latex_code = reflection_code_match.group(1).strip()
            if reflected_latex_code != current_latex:
                final_text = reflected_latex_code
                cleanup_map = {
                    "</end": r"\\end",
                    "</begin": r"\\begin",
                    "’": "'",
                }
                for bad_str, repl_str in cleanup_map.items():
                    final_text = final_text.replace(bad_str, repl_str)
                final_text = re.sub(r"(\d+(?:\.\d+)?)%", r"\1\\%", final_text)

                with open(writeup_file, "w") as fo:
                    fo.write(final_text)

                compile_latex(latex_folder, reflection_pdf)
            else:
                print(f"No changes in reflection page step.")

        return osp.exists(reflection_pdf)

    except Exception:
        print("EXCEPTION in perform_writeup:")
        print(traceback.format_exc())
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform writeup for a project")
    parser.add_argument("--folder", type=str, help="Project folder", required=True)
    parser.add_argument("--no-writing", action="store_true", help="Only generate")
    parser.add_argument("--num-cite-rounds", type=int, default=20)
    parser.add_argument(
        "--model",
        type=str,
        default="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
        help="Model to use for citation collection (small model).",
    )
    parser.add_argument(
        "--big-model",
        type=str,
        default="o1-2024-12-17",
        help="Model to use for final writeup (big model).",
    )
    parser.add_argument(
        "--writeup-reflections",
        type=int,
        default=3,
        help="Number of reflection steps for the final LaTeX writeup.",
    )
    parser.add_argument(
        "--page-limit",
        type=int,
        default=4,
        help="Target page limit for the main paper (excluding references).",
    )
    args = parser.parse_args()

    try:
        success = perform_writeup(
            base_folder=args.folder,
            no_writing=args.no_writing,
            num_cite_rounds=args.num_cite_rounds,
            small_model=args.model,
            big_model=args.big_model,
            n_writeup_reflections=args.writeup_reflections,
            page_limit=args.page_limit,
        )
        if not success:
            print("Writeup process did not complete successfully.")
    except Exception:
        print("EXCEPTION in main:")
        print(traceback.format_exc())
