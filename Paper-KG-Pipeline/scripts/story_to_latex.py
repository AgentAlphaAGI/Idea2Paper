#!/usr/bin/env python3
"""
将 final_story.json 填充到 arXiv LaTeX 模板并输出 .tex 文件。

Usage (在 Idea2Paper 仓库根目录执行):
  python Paper-KG-Pipeline/scripts/story_to_latex.py <results_dir>
  python Paper-KG-Pipeline/scripts/story_to_latex.py "results/Lite Digital Twin"
  python Paper-KG-Pipeline/scripts/story_to_latex.py Paper-KG-Pipeline/output

  --no-download  不下载 arxiv.sty，使用纯标准 article 模板（离线可用）
  -o paper.tex   指定输出文件名

功能:
  1. 读取指定 results 目录下的 final_story.json
  2. 自动下载 arXiv LaTeX 模板 (kourgeorge/arxiv-style)
  3. 将 story 内容填充到模板
  4. 输出 paper.tex 到 final_story.json 所在目录
"""

import argparse
import json
import re
import sys
from pathlib import Path

try:
    import urllib.request
    from urllib.error import URLError, HTTPError
except ImportError:
    urllib = None

# arXiv 模板下载地址 (kourgeorge/arxiv-style)
ARXIV_TEMPLATE_URL = "https://raw.githubusercontent.com/kourgeorge/arxiv-style/master/template.tex"
ARXIV_STY_URL = "https://raw.githubusercontent.com/kourgeorge/arxiv-style/master/arxiv.sty"


def _latex_escape(text: str) -> str:
    """转义 LaTeX 特殊字符"""
    if not text:
        return ""
    # 顺序重要：先处理反斜杠
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def _download_url(url: str) -> str:
    """下载 URL 内容，返回文本"""
    req = urllib.request.Request(url, headers={"User-Agent": "Idea2Paper/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8")


def _parse_method_steps(text: str) -> list[str]:
    """将 method_skeleton 解析为步骤列表"""
    if not text or not text.strip():
        return []
    # 匹配 "Step N: ..." 或 "Step N. ..."
    steps = re.split(r"\s*Step\s+\d+\s*[.:]\s*", text, flags=re.IGNORECASE)
    steps = [s.strip() for s in steps if s.strip()]
    if not steps and text.strip():
        # 无 Step 前缀时按分号分割
        steps = [s.strip() for s in text.split(";") if s.strip()]
    return steps


def _build_arxiv_template(use_arxiv_sty: bool = True) -> str:
    """构建用于填充的 arXiv 风格模板（含占位符）"""
    if use_arxiv_sty:
        preamble = r"""\documentclass{article}

\usepackage{arxiv}

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{hyperref}
\usepackage{url}
\usepackage{booktabs}
\usepackage{amsfonts}
\usepackage{graphicx}
\usepackage{natbib}
\usepackage{doi}

\title{__TITLE__}

\author{Anonymous Author(s)\thanks{Replace with your affiliation and email.}}

\renewcommand{\shorttitle}{\textit{arXiv} Preprint}
"""
    else:
        preamble = r"""\documentclass{article}

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{hyperref}
\usepackage{url}
\usepackage{booktabs}
\usepackage{amsfonts}
\usepackage{graphicx}

\title{__TITLE__}

\author{Anonymous Author(s)\thanks{Replace with your affiliation and email.}}
"""
    return preamble + r"""

\hypersetup{
pdftitle={__TITLE__},
pdfauthor={Anonymous},
pdfkeywords={Research, Paper},
}

\begin{document}
\maketitle

\begin{abstract}
__ABSTRACT__
\end{abstract}

\section{Introduction}
\label{sec:intro}

__PROBLEM_FRAMING__

__GAP_PATTERN__

\section{Methodology}
\label{sec:method}

__SOLUTION__

\subsection{Technical Steps}

__METHOD_SKELETON__

\section{Contributions}
\label{sec:contributions}

__INNOVATION_CLAIMS__

\section{Experiments}
\label{sec:experiments}

__EXPERIMENTS_PLAN__

\end{document}
"""


def _fill_template(template: str, story: dict) -> str:
    """将 story 内容填充到模板"""
    title = _latex_escape(story.get("title", ""))
    abstract = _latex_escape(story.get("abstract", ""))
    problem_framing = _latex_escape(story.get("problem_framing", ""))
    gap_pattern = _latex_escape(story.get("gap_pattern", ""))
    solution = _latex_escape(story.get("solution", ""))
    experiments_plan = _latex_escape(story.get("experiments_plan", ""))

    # method_skeleton -> itemize
    method_text = story.get("method_skeleton", "")
    steps = _parse_method_steps(method_text)
    if steps:
        method_latex = "\\begin{itemize}\n"
        for s in steps:
            method_latex += f"  \\item {_latex_escape(s)}\n"
        method_latex += "\\end{itemize}"
    else:
        method_latex = _latex_escape(method_text) if method_text else ""

    # innovation_claims -> itemize
    claims = story.get("innovation_claims", [])
    if isinstance(claims, list):
        claims_latex = "\\begin{itemize}\n"
        for c in claims:
            claims_latex += f"  \\item {_latex_escape(str(c))}\n"
        claims_latex += "\\end{itemize}"
    else:
        claims_latex = _latex_escape(str(claims))

    replacements = {
        "__TITLE__": title,
        "__ABSTRACT__": abstract,
        "__PROBLEM_FRAMING__": problem_framing,
        "__GAP_PATTERN__": gap_pattern,
        "__SOLUTION__": solution,
        "__METHOD_SKELETON__": method_latex,
        "__INNOVATION_CLAIMS__": claims_latex,
        "__EXPERIMENTS_PLAN__": experiments_plan,
    }
    result = template
    for k, v in replacements.items():
        result = result.replace(k, v)
    return result


def _download_arxiv_template(output_dir: Path) -> str:
    """下载 arXiv 模板并返回填充用模板内容。下载失败时回退到内置模板（无需 arxiv.sty）。"""
    try:
        template_content = _download_url(ARXIV_TEMPLATE_URL)
        sty_content = _download_url(ARXIV_STY_URL)
    except (URLError, HTTPError, OSError) as e:
        print(f"[warn] 无法下载 arXiv 模板: {e}", file=sys.stderr)
        return _build_arxiv_template(use_arxiv_sty=False)

    # 保存 arxiv.sty 到输出目录（模板需要）
    sty_path = output_dir / "arxiv.sty"
    sty_path.write_text(sty_content, encoding="utf-8")
    print(f"  ✓ 已保存 arxiv.sty 到 {output_dir}")

    # 从下载的模板提取结构，替换为我们的占位符
    # 简化：使用内置模板结构，但保留 arxiv 包
    return _build_arxiv_template()


def main():
    parser = argparse.ArgumentParser(
        description="将 final_story.json 填充到 arXiv LaTeX 模板并输出 .tex 文件"
    )
    parser.add_argument(
        "results_dir",
        type=str,
        help="包含 final_story.json 的 results 目录路径",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="paper.tex",
        help="输出 .tex 文件名 (默认: paper.tex)",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="不下载 arXiv 模板，仅使用内置模板",
    )
    args = parser.parse_args()

    results_path = Path(args.results_dir).resolve()
    story_file = results_path / "final_story.json"

    if not story_file.exists():
        print(f"错误: 未找到 {story_file}", file=sys.stderr)
        sys.exit(1)

    print(f"读取: {story_file}")
    with story_file.open("r", encoding="utf-8") as f:
        story = json.load(f)

    # 下载或使用内置模板
    if args.no_download:
        template = _build_arxiv_template(use_arxiv_sty=False)
        print("使用内置模板（无需 arxiv.sty）")
    else:
        print("下载 arXiv LaTeX 模板...")
        template = _download_arxiv_template(results_path)

    # 填充并输出
    filled = _fill_template(template, story)
    output_path = results_path / args.output
    output_path.write_text(filled, encoding="utf-8")

    print(f"✓ 已输出 LaTeX 文件: {output_path}")
    print(f"  编译: cd {results_path} && pdflatex paper.tex")


if __name__ == "__main__":
    main()
