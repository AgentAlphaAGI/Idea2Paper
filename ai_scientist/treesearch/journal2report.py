from .backend import query
from .journal import Journal
from .utils.config import StageConfig
from ai_scientist.prompts.loader import load_prompt, render_prompt


def journal2report(journal: Journal, task_desc: dict, rcfg: StageConfig):
    """
    Generate a report from a journal, the report will be in markdown format.
    """
    report_input = journal.generate_summary(include_code=True)
    system_prompt_dict = load_prompt("journal2report.system_prompt")
    context_prompt = render_prompt(
        "journal2report.context_prompt",
        report_input=report_input,
        task_desc=task_desc,
    )
    return query(
        system_message=system_prompt_dict,
        user_message=context_prompt,
        model=rcfg.model,
        temperature=rcfg.temp,
        max_tokens=rcfg.max_tokens,
    )
