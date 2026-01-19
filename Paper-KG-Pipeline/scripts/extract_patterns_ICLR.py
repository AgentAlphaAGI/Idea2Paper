# extract_patterns_100.py
import os, json, time
from typing import Any, Dict, List
from datasets import load_dataset
from tqdm import tqdm

from openai import OpenAI

DATASET_NAME = "AgentAlphaAGI/Paper-Review-Dataset"
SPLIT = "train"
N = 10


MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")  # 你可改成 gpt-4.1 / gpt-4o 等
OUT_PATH = "patterns_100.jsonl"

# ====== 1) 把上面那段“增强版 Prompt”粘到这里 ======
PROMPT_TEMPLATE = r"""
【任务目标】
基于输入的论文标题、关键词和摘要，精准提取顶会级论文的“Research Pattern 核心框架”，用于学习顶级论文如何把“朴素方法”包装成“有理论深度+实验说服力+想象空间”的可复用叙事套路。

【核心概念定义】
1) Research Pattern = ⟨Base Problem, Solution Pattern, Story⟩
- Base Problem：论文在“具体应用场景”要解决的“核心痛点”，要求具体、可落地，避免空泛。
- Solution Pattern：解决 Base Problem 的“核心技术路径”，说清关键组件、流程结构、机制与相互关系。
- Story：论文对 Solution Pattern 的“概念化包装/叙事框架”，强调前瞻性、概念创新性、价值想象空间、领域引领性（这是最重要的字段）。

2) 其他字段
- domains：论文所属一级/二级技术领域（可多个），必须与论文强相关。
- application：Solution Pattern 可落地的具体业务/技术场景，结合论文内容写清楚。
- idea: 可选字段，论文提出的关键创新点/核心思路，简洁描述, 一两句话。

【输出要求（非常重要）】
- 只输出 JSON，禁止任何额外解释、前后缀、Markdown、项目符号。
- 所有字段都用中文表述（学术中文，简洁但信息密度高）。
- research_patterns 至少 1 个对象，每个对象内字段不可为空。
- paper_id 必须填：{paper_info['paper_id']}
- paper_title 必须填输入标题；无则填“无”
- domains 必须是数组；无明确领域则填 ["待明确领域"]

【输出格式】
{
  "paper_id": "{paper_info['paper_id']}",
  "paper_title": "论文标题",
  "domains": ["领域1", "领域2"],
  "idea": "论文的关键创新点/核心思路（可选）",
  "research_patterns": [
    {
      "base_problem": "具体场景下的核心痛点（精准对应论文研究目标）",
      "solution_pattern": "核心技术方案（组件/流程/机制/创新点）",
      "story": "论文的概念化包装（前瞻性+想象空间+领域引领）",
      "application": "可落地的业务或技术场景"
    }
  ]
}

========================
【Few-shot 示例 1】
【输入】
- 论文标题: 基于 Reflect+Memory 的智能 Agent 自进化研究
- 关键词: 智能Agent, 自进化, 记忆机制
- 摘要: 现有智能 Agent 执行任务后无法留存经验，重复执行同类任务时易重复踩坑，效率低下。本文提出在 Agent 基础架构中引入 Reflect 反思模块与 Memory 长效记忆模块，实现经验的留存、总结与复用，提升 Agent 任务执行效率。Agent 从此学会思考，可以不断总结经验，随着经验的积累，Agent 会越来越强大，这是一个能自我学习的 agent。

【输出】
{
  "paper_id": "ARR_2022_106",
  "paper_title": "基于 Reflect+Memory 的智能 Agent 自进化研究",
  "domains": ["人工智能", "智能体系统", "自主学习"],
  “idea”: "提出“Reflect+Memory”架构：通过反思模块总结经验并存入长效记忆，实现智能体的经验积累与持续改进",
  "research_patterns": [
    {
      "base_problem": "智能体在完成任务后无法沉淀可复用经验，导致同类任务重复犯错、效率低且难以持续变强",
      "solution_pattern": "在智能体架构中加入反思模块对轨迹进行归因总结，并将结构化经验写入长效记忆；推理时检索相似经验注入决策，从而实现经验继承与持续改进",
      "story": "提出“可自我进化的智能体范式”：让 Agent 像研究者一样反思并积累经验库，实现从一次性执行到持续学习的跃迁，打开长期自主能力增长的想象空间",
      "application": "智能客服自动复盘与话术迭代、运维/故障处理自动总结、具身机器人任务技能迁移、自动驾驶决策策略持续优化"
    }
  ]
}

【Few-shot 示例 2】
【输入】
- 论文标题: 面向长视频异常检测的多粒度层级推理框架
- 关键词: 异常检测, 长视频, 多粒度, 层级推理
- 摘要: 现有方法难以同时覆盖短时与长时异常，缺少跨时间粒度的证据聚合机制。本文构建多层级时间树表示，将视频切分为多尺度片段并进行局部评分，再通过层级融合实现全局异常定位与解释。

【输出】
{
  "paper_id": "VAD_2024_013",
  "paper_title": "面向长视频异常检测的多粒度层级推理框架",
  "domains": ["计算机视觉", "视频理解", "异常检测"],
  "idea": "提出多尺度时间层级树结构：通过多粒度片段表征与局部评分，结合层级聚合与跨尺度融合，实现对长视频中短时突发与长时演化异常的精准检测与解释",
  "research_patterns": [
    {
      "base_problem": "长视频异常事件跨度不一且证据分散，单一时间尺度建模无法同时捕捉短时突发与长时缓慢演化异常，导致漏检与不可解释",
      "solution_pattern": "构建多尺度时间层级树：在不同粒度对片段进行表征与局部打分，再通过层级聚合与跨尺度融合实现全局异常评分、定位与证据回溯",
      "story": "将异常检测重述为“跨粒度证据推理”问题：用层级结构把碎片化信号组织成可推理的证据链，实现从黑盒打分到可解释定位的范式升级",
      "application": "安防监控长时段异常巡检、工业产线视频质检、医疗监护异常预警、交通流异常检测与溯源"
    }
  ]
}

========================
【现在开始抽取（真实输入）】
- 论文标题: {paper_info['paper_title']}
- 关键词: {paper_info['keywords']}
- 摘要: {paper_info['abstract']}

请严格按 JSON 输出（中文），不得输出任何其他内容。

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


def call_llm(client: OpenAI, prompt: str) -> Dict[str, Any]:
    # 关键点：让模型“只输出 JSON”
    # 兼容性写法：用 response_format={"type":"json_object"} 强约束（如你的 SDK/模型支持）
    resp = client.responses.create(
    model=MODEL,
    input=[{"role": "user", "content": prompt}],
    temperature=0.2,
    )

    text = resp.output_text
    return json.loads(text)


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
