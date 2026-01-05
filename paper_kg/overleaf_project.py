# 中文注释: 本文件负责生成 Overleaf/LaTeX 工程目录与 zip 包。
from __future__ import annotations

import json
import logging
import re
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from paper_kg.latex_utils import normalize_latex_newlines
from paper_kg.models import PaperPlan, PaperSectionOutput


logger = logging.getLogger(__name__)


def build_overleaf_project(
    run_id: str,
    output_dir: str,
    plan: PaperPlan,
    sections: List[PaperSectionOutput],
    refs_bib_text: str | None = None,
    citations_report: Dict | None = None,
    latex_template_preset: str = "article_ctex",
    latex_titlepage_style: str = "qwen",
    latex_title: str = "",
    latex_author: str = "",
    latex_hflink: str = "https://huggingface.co/Qwen",
    latex_mslink: str = "https://modelscope.cn/organization/qwen",
    latex_ghlink: str = "https://github.com/QwenLM/Qwen3",
    latex_template_assets_dir: str = "arXiv-2505.09388v1",
) -> Dict[str, str]:
    """
    功能：构建 Overleaf 工程目录与 zip 包。
    参数：run_id/output_dir/plan/sections。
    返回：包含 project_dir/zip_path 的 dict。
    流程：创建目录 → 写 main.tex/章节 tex → 写 refs.bib → 生成 zip。
    说明：main.tex 由系统固定生成，LLM 不参与导言区。
    """
    project_dir = Path(output_dir) / run_id
    sections_dir = project_dir / "sections"
    sections_dir.mkdir(parents=True, exist_ok=True)

    preset = (latex_template_preset or "article_ctex").strip().lower()
    titlepage_style = (latex_titlepage_style or "qwen").strip().lower()
    if preset == "colm2024_conference":
        _copy_colm_assets(project_dir, Path(latex_template_assets_dir))
        if titlepage_style == "agentalpha":
            _copy_agentalpha_logo(project_dir)

    section_map = {section.section_id: section for section in sections}
    for section_plan in plan.sections:
        section = section_map.get(section_plan.section_id)
        tex_content = section.content_latex if section else ""
        if not tex_content.strip():
            tex_content = _placeholder_section(section_plan.section_id, section_plan.title)
        tex_content = normalize_latex_newlines(tex_content)
        (sections_dir / f"{section_plan.section_id}.tex").write_text(
            tex_content,
            encoding="utf-8",
        )

    main_tex = _render_main_tex(
        plan,
        latex_template_preset=preset,
        latex_titlepage_style=titlepage_style,
        latex_title=latex_title,
        latex_author=latex_author,
        latex_hflink=latex_hflink,
        latex_mslink=latex_mslink,
        latex_ghlink=latex_ghlink,
    )
    (project_dir / "main.tex").write_text(main_tex, encoding="utf-8")

    bib_path = project_dir / "refs.bib"
    if refs_bib_text and refs_bib_text.strip():
        bib_path.write_text(refs_bib_text, encoding="utf-8")
    else:
        cite_keys = _collect_cite_keys(sections_dir)
        bib_text = _build_bib_entries(cite_keys)
        bib_path.write_text(bib_text, encoding="utf-8")

    if citations_report:
        report_path = project_dir / "citations_report.json"
        report_path.write_text(json.dumps(citations_report, ensure_ascii=False, indent=2), encoding="utf-8")

    zip_path = project_dir / "overleaf.zip"
    _zip_project(project_dir, zip_path)

    logger.info("OVERLEAF_PROJECT_OK: dir=%s", project_dir)
    return {"project_dir": str(project_dir), "zip_path": str(zip_path)}


def _render_main_tex(
    plan: PaperPlan,
    latex_template_preset: str = "article_ctex",
    latex_titlepage_style: str = "qwen",
    latex_title: str = "",
    latex_author: str = "",
    latex_hflink: str = "https://huggingface.co/Qwen",
    latex_mslink: str = "https://modelscope.cn/organization/qwen",
    latex_ghlink: str = "https://github.com/QwenLM/Qwen3",
) -> str:
    """
    功能：渲染固定 main.tex 模板。
    参数：plan。
    返回：main.tex 文本。
    说明：模板固定，避免 LLM 改写导言区。
    """
    section_inputs = []
    preset = (latex_template_preset or "article_ctex").strip().lower()
    titlepage_style = (latex_titlepage_style or "qwen").strip().lower()
    for section_plan in plan.sections:
        if preset == "colm2024_conference" and section_plan.section_id == "abstract":
            continue
        section_inputs.append(f"\\input{{sections/{section_plan.section_id}.tex}}")
    inputs_text = "\n".join(section_inputs)

    template_root = Path(__file__).resolve().parent / "latex_templates"
    if preset == "colm2024_conference":
        template_name = "main.tex.template"
        if titlepage_style == "agentalpha":
            template_name = "main_agentalpha.tex.template"
        template_path = template_root / "colm2024_conference" / template_name
        if not template_path.exists():
            raise RuntimeError(f"missing COLM template file: {template_path}")
        template = template_path.read_text(encoding="utf-8")
        title = latex_title.strip() or "TODO Title"
        author = latex_author.strip() or "TODO Authors"
        return (
            template.replace("%%SECTION_INPUTS%%", inputs_text)
            .replace("%%TITLE%%", title)
            .replace("%%AUTHOR%%", author)
            .replace("%%HFLINK%%", latex_hflink)
            .replace("%%MSLINK%%", latex_mslink)
            .replace("%%GHLINK%%", latex_ghlink)
        )

    template_path = template_root / "article_ctex" / "main.tex.template"
    if template_path.exists():
        template = template_path.read_text(encoding="utf-8")
        return template.replace("%%SECTION_INPUTS%%", inputs_text)

    # 中文注释：兜底模板（避免模板文件缺失导致失败）。
    return (
        "\\documentclass{article}\n"
        "\\usepackage{ctex}\n"
        "\\usepackage{amsmath}\n"
        "\\usepackage{amssymb}\n"
        "\\usepackage{graphicx}\n"
        "% --- Safe includegraphics: do not fail compilation if figure files are missing ---\n"
        "\\let\\origincludegraphics\\includegraphics\n"
        "\\newcommand{\\MissingFigureBox}[1]{%\n"
        "  \\fbox{\\parbox[c][0.18\\textheight][c]{0.9\\linewidth}{\\centering Missing figure: \\texttt{#1}}}%\n"
        "}\n"
        "\\renewcommand{\\includegraphics}[2][]{%\n"
        "  \\IfFileExists{#2}{\\origincludegraphics[#1]{#2}}{%\n"
        "    \\IfFileExists{#2.pdf}{\\origincludegraphics[#1]{#2.pdf}}{%\n"
        "      \\IfFileExists{#2.png}{\\origincludegraphics[#1]{#2.png}}{%\n"
        "        \\IfFileExists{#2.jpg}{\\origincludegraphics[#1]{#2.jpg}}{%\n"
        "          \\IfFileExists{#2.jpeg}{\\origincludegraphics[#1]{#2.jpeg}}{%\n"
        "            \\MissingFigureBox{\\detokenize{#2}}%\n"
        "          }%\n"
        "        }%\n"
        "      }%\n"
        "    }%\n"
        "  }%\n"
        "}\n"
        "% -------------------------------------------------------------------------------\n"
        "\\usepackage{booktabs}\n"
        "\\usepackage{longtable}\n"
        "\\usepackage{array}\n"
        "\\usepackage{natbib}\n"
        "\\providecommand{\\tightlist}{}\n"
        "\\begin{document}\n"
        f"{inputs_text}\n"
        "\\bibliographystyle{plainnat}\n"
        "\\bibliography{refs}\n"
        "\\end{document}\n"
    )


def _collect_cite_keys(sections_dir: Path) -> List[str]:
    """
    功能：扫描章节 tex 收集 cite keys。
    参数：sections_dir。
    返回：key 列表（去重）。
    说明：支持 \\cite/\\citep/\\citet 等形式。
    """
    cite_pattern = re.compile(r"\\cite[a-zA-Z]*\{([^}]*)\}")
    keys: List[str] = []
    for tex_file in sections_dir.glob("*.tex"):
        content = tex_file.read_text(encoding="utf-8")
        for match in cite_pattern.finditer(content):
            raw_keys = match.group(1)
            for key in raw_keys.split(","):
                key = key.strip()
                if key:
                    keys.append(key)
    return sorted(set(keys))


def _build_bib_entries(keys: Iterable[str]) -> str:
    """
    功能：为 cite keys 生成占位 BibTeX 条目。
    参数：keys。
    返回：refs.bib 文本。
    说明：保证 bibkey 永不缺失。
    """
    entries = []
    for key in keys:
        safe_key = _sanitize_bib_key(key)
        entries.append(
            (
                f"@misc{{{safe_key},\n"
                "  title = {TODO Placeholder},\n"
                "  year = {2026},\n"
                "  note = {Auto-generated placeholder}\n"
                "}\n"
            )
        )
    return "\n".join(entries)


def _sanitize_bib_key(key: str) -> str:
    """
    功能：清理 bibkey 的非法字符。
    参数：key。
    返回：安全 key。
    """
    return re.sub(r"[^A-Za-z0-9:_-]+", "", key)


def _zip_project(project_dir: Path, zip_path: Path) -> None:
    """
    功能：将 Overleaf 工程目录打包为 zip。
    参数：project_dir/zip_path。
    返回：无。
    说明：忽略 zip 自身。
    """
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        for path in project_dir.rglob("*"):
            if path == zip_path:
                continue
            if path.is_file():
                zipf.write(path, path.relative_to(project_dir))


def _placeholder_section(section_id: str, title: str) -> str:
    """
    功能：生成可编译的占位章节内容。
    参数：section_id/title。
    返回：LaTeX 文本。
    说明：用于最终兜底。
    """
    if section_id == "abstract":
        return "TODO: 内容待补充。\n"
    return f"\\section{{{title or section_id}}}\nTODO: 内容待补充。\n"


def _copy_colm_assets(project_dir: Path, assets_dir: Path) -> None:
    assets_dir = Path(assets_dir).expanduser()
    if not assets_dir.exists():
        raise RuntimeError(f"missing COLM assets dir: {assets_dir}")

    required_files = [
        "colm2024_conference.sty",
        "colm2024_conference.bst",
    ]
    for filename in required_files:
        src = assets_dir / filename
        if not src.exists():
            raise RuntimeError(f"missing COLM asset: {src}")
        shutil.copy2(src, project_dir / filename)

    logo_dir = assets_dir / "logo"
    if not logo_dir.exists():
        raise RuntimeError(f"missing COLM logo dir: {logo_dir}")

    required_logos = [
        "qwen-logo.pdf",
        "hf-logo.pdf",
        "modelscope-logo.pdf",
        "github-logo.pdf",
    ]
    for logo in required_logos:
        if not (logo_dir / logo).exists():
            raise RuntimeError(f"missing COLM logo: {logo_dir / logo}")

    shutil.copytree(logo_dir, project_dir / "logo", dirs_exist_ok=True)


def _copy_agentalpha_logo(project_dir: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "AgentAlpha_template" / "logo.png"
    if not src.exists():
        raise RuntimeError(f"missing AgentAlpha logo: {src}")
    logo_dir = project_dir / "logo"
    logo_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, logo_dir / "agentalpha-logo.png")
