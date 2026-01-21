# extract_patterns_100.py
import os, json, time
from typing import Any, Dict, List
from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path

from openai import OpenAI

DATASET_NAME = "AgentAlphaAGI/Paper-Review-Dataset"
SPLIT = "train"
N = 8310


MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")  # 你可改成 gpt-4.1 / gpt-4o 等

# 获取项目根目录 (知识图谱Pipeline)
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
# 输出路径
OUTPUT_DIR = PROJECT_ROOT / "output"
OUT_PATH = OUTPUT_DIR / "iclr_patterns_full_cn.jsonl"

# ====== 1) 把上面那段“增强版 Prompt”粘到这里 ======
PROMPT_TEMPLATE = r"""
【角色】
你是一名“顶会论文 Research Pattern 抽取专家”。
你的任务不是复述或摘要论文内容，而是抽取**顶级论文如何将看似普通的方法，通过问题重定义、系统化组织与概念化叙事，包装为高影响力研究工作的可复用框架**。

【任务目标】
基于输入的论文标题、关键词和摘要，抽取论文的 Research Pattern 核心结构：
⟨Base Problem, Solution Pattern, Story⟩，
并识别论文的**核心创新思路（idea）**。

该抽取用于学习顶会论文的“研究叙事方式”。其中 **Story 是最重要的字段**。

---

【核心概念定义】

### 1) Research Pattern = ⟨Base Problem, Solution Pattern, Story⟩

- **Base Problem（基础问题）**  
  论文在**具体研究或应用场景**中要解决的**核心痛点**。  
  要求明确说明：  
  - 在什么场景下  
  - 出现了什么关键问题  
  - 为什么现有方法不够或不可靠  
  禁止宽泛描述（如“提升性能”“效果不好”）。

- **Solution Pattern（解决方案套路）**  
  用于解决 Base Problem 的**核心技术路径**。  
  要求描述清楚：  
  - 关键技术组件  
  - 整体流程或结构  
  - 方法是如何通过系统性组织产生效果的  
  而不是实现细节或参数设置。

- **Story（叙事与概念包装，最重要）**  
  论文对 Solution Pattern 的**概念化叙事方式**。  
  关注论文如何：
  - 重新定义问题或研究对象  
  - 将工程问题上升为研究命题（如可靠性、公平性、审计、风险等）  
  - 强调前瞻性、研究范式意义或长期影响力  
  该字段用于学习**顶会论文是如何“讲故事”的**，不得只是技术总结。

---

### 2) idea（论文级，可选）

- 用 **1–2 句话**概括论文提出的**关键创新点或核心思路**；
- 关注“这篇论文本质上新在哪里”，不需要技术细节；
- 若无法清晰表达，可省略该字段。

---

### 3) 领域标签（用于检索与召回）

- **domain（一级领域，字符串）**  
  论文所属的主要研究领域，应为较宽泛的学科标签。  
  示例：
  - “机器学习”
  - “计算机视觉”
  - “自然语言处理”
  - “系统”
  - “安全与隐私”
  - “公平性与可信 AI”

- **sub_domains（二级领域，字符串数组）**  
  domain 下更具体、可用于精细召回的子方向，建议 2–5 个。  
  示例：
  - “联邦学习”
  - “标签噪声”
  - “影响函数”
  - “层级建模”
  - “异常检测”
  
  规则：
  - sub_domains 必须与 domain 语义一致  
  - 不要重复 domain 本身

---

### 4) application（应用场景）

- Solution Pattern 在现实中**可落地的具体业务或技术应用场景**。

---

【输出要求（严格）】
1. **仅输出 JSON**，不允许任何额外说明、注释或 Markdown；
2. 所有 value 必须使用**中文**，语言专业、凝练、符合学术语境；
3. 字段约束：
   - paper_id 必须填：{paper_info['paper_id']}
   - paper_title 使用输入标题；无则填“无”
   - idea 为可选字段
   - domain 必须为非空字符串
   - sub_domains 必须为非空数组（至少 1 项）
   - research_patterns 必须为非空数组（至少 1 个对象）
   - research_patterns 内部字段不得为空

---

【输出格式】
{
  "paper_id": "{paper_info['paper_id']}",
  "paper_title": "论文标题",
  "idea": "论文的核心创新点或关键思路（可选，1–2 句话）",
  "domain": "一级领域",
  "sub_domains": ["二级领域1", "二级领域2"],
  "research_patterns": [
    {
      "base_problem": "具体研究或应用场景下的核心痛点",
      "solution_pattern": "核心技术方案与流程结构",
      "story": "论文的概念化叙事与前瞻性包装方式",
      "application": "可落地的具体应用场景"
    }
  ]
}

--------------------------------------------------
【Few-Shot 示例一】

【输入】
- 论文标题：基于 Reflect+Memory 的智能 Agent 自进化研究
- 关键词：智能 Agent，自进化，记忆机制
- 摘要：
  现有智能 Agent 在完成任务后无法系统性沉淀经验，
  在相似任务中容易重复犯错，长期能力提升受限。
  本文在 Agent 架构中引入 Reflect 反思模块与 Memory 长期记忆模块，
  实现经验的总结、存储与复用，使 Agent 能在持续交互中不断进化。

【输出】
{
  "paper_id": "ARR_2022_106",
  "paper_title": "基于 Reflect+Memory 的智能 Agent 自进化研究",
  "idea": "通过反思与长期记忆机制，使智能体能够沉淀和复用经验，实现持续自我进化",
  "domain": "人工智能",
  "sub_domains": ["智能体系统", "记忆增强模型", "反思机制", "经验检索"],
  "research_patterns": [
    {
      "base_problem": "在持续执行任务的场景下，智能 Agent 无法有效沉淀和复用历史经验，导致在相似任务中反复犯错，长期能力提升受限",
      "solution_pattern": "在智能体架构中引入反思模块对任务轨迹进行总结，并将结构化经验写入长期记忆，在推理阶段检索相关经验指导决策",
      "story": "将智能体从一次性任务执行者重塑为可持续学习的自进化系统，构建以经验积累驱动能力增长的研究范式",
      "application": "智能客服持续优化、自动化运维经验沉淀、机器人技能迁移、长期决策辅助系统"
    }
  ]
}

--------------------------------------------------
【Few-Shot 示例二】

【输入】
- 论文标题：Quantifying and Mitigating the Impact of Label Errors on Model Disparity Metrics
- 关键词：公平性，标签噪声，影响函数，差异指标
- 摘要：
  现有公平性评估方法通常假设标签可靠，
  但现实数据中标签错误普遍存在，并对不同群体产生不均衡影响。
  本文分析标签错误对群体差异指标的影响，
  并提出基于影响函数的方法定位并修正高影响错误标签。

【输出】
{
  "paper_id": "ICLR_2023_089",
  "paper_title": "Quantifying and Mitigating the Impact of Label Errors on Model Disparity Metrics",
  "idea": "通过分析单个标签错误对群体差异指标的影响，提升公平性评估与缓解的可靠性",
  "domain": "公平性与可信人工智能",
  "sub_domains": ["标签噪声", "影响函数", "公平性评估", "模型审计"],
  "research_patterns": [
    {
      "base_problem": "在存在标签噪声的现实数据中，公平性评估默认标签可靠，导致群体差异指标被系统性扭曲，尤其影响少数群体的公平判断",
      "solution_pattern": "将影响函数从样本损失分析扩展至群体差异指标，量化单个标签扰动对公平性的影响，并优先修正高影响错误标签",
      "story": "将公平性问题从模型优化层面提升为评估可靠性与数据可信度问题，引入公平性审计视角，构建可诊断、可干预的系统性分析框架",
      "application": "高风险决策系统的公平性审计、噪声数据下的公平性评估与纠偏、自动化数据质量与偏差检测"
    }
  ]
}

--------------------------------------------------
【正式输入】
- 论文标题：{paper_info['paper_title']}
- 关键词：{paper_info['keywords']}
- 摘要：{paper_info['abstract']}

【输出】
仅输出符合上述格式的 JSON。

"""


def build_paper_info(row: Dict[str, Any]) -> Dict[str, Any]:
    # HF 数据集字段：title/authors/abstract/pdf_url/source_url/id/related_notes/year/conference/content/content_meta
    # keywords：数据集中没有单独列，这里先留空数组（后续你可加 keyphrase 模块自动生成）
    return {
        "paper_id": row.get("id", "") or "",
        "paper_title": row.get("title", "") or "无",
        "keywords": [],  # <- 先空
        "abstract": (row.get("abstract", "") or "").strip(),
        "source_url": row.get("source_url", ""),
        "pdf_url": row.get("pdf_url", ""),
        "year": str(row.get("year", "")),
        "conference": row.get("conference", ""),
    }


def render_prompt(paper_info: Dict[str, Any]) -> str:
    # 这里用最简单的字符串替换；如果你担心 prompt 里有花括号冲突，可用更稳健的模板引擎
    prompt = PROMPT_TEMPLATE
    prompt = prompt.replace("{paper_info['paper_id']}", paper_info["paper_id"])
    prompt = prompt.replace("{paper_info['paper_title']}", paper_info["paper_title"])
    prompt = prompt.replace("{paper_info['keywords']}", ", ".join(paper_info["keywords"]) if paper_info["keywords"] else "无")
    prompt = prompt.replace("{paper_info['abstract']}", paper_info["abstract"])
    return prompt


def call_llm(client: OpenAI, prompt: str) -> dict:
    resp = client.responses.create(
        model=MODEL,
        input=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    text = resp.output_text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 兜底：抽取 JSON 主体
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            return json.loads(text[start:end+1])
        raise


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY env var")

    client = OpenAI(api_key=api_key)

    ds = load_dataset(DATASET_NAME, split=SPLIT)  # 默认会从 HF 拉取 json 数据
    rows = ds.select(range(min(N, len(ds))))

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for row in tqdm(rows, total=len(rows), desc="Extracting patterns"):
            paper_info = build_paper_info(row)
            prompt = render_prompt(paper_info)

            # 简单重试 + 节流
            for attempt in range(3):
                try:
                    obj = call_llm(client, prompt)
                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    f.flush()
                    break
                except Exception as e:
                    if attempt == 2:
                        # 写一条错误记录，方便你后处理补跑
                        err = {
                            "paper_id": paper_info["paper_id"],
                            "paper_title": paper_info["paper_title"],
                            "error": str(e),
                        }
                        f.write(json.dumps(err, ensure_ascii=False) + "\n")
                        f.flush()
                    time.sleep(2.0 * (attempt + 1))


if __name__ == "__main__":
    main()
