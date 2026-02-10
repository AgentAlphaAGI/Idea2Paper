"""
Idea2Story Pipeline - 从用户 Idea 到可发表的 Paper Story

实现流程:
  Phase 1: Pattern Selection (策略选择)
  Phase 2: Story Generation (结构化生成)
  Phase 3: Multi-Agent Critic & Refine (评审与修正)
  Phase 4: RAG Verification & Pivot (查重与规避)

使用方法:
  python scripts/idea2story_pipeline.py "你的Idea描述"
"""

import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

# 提前加载 .env（确保 PipelineConfig 读取前生效）
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
REPO_ROOT = PROJECT_ROOT.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from idea2paper.infra.dotenv import load_dotenv
    _DOTENV_STATUS = load_dotenv(REPO_ROOT / ".env", override=False)
except Exception as _e:
    _DOTENV_STATUS = {"loaded": 0, "path": str(REPO_ROOT / ".env"), "ok": False, "error": str(_e)}

# 导入 Pipeline 模块
try:
    from pipeline import Idea2StoryPipeline, OUTPUT_DIR
    from pipeline.config import (
        LOG_ROOT,
        ENABLE_RUN_LOGGING,
        LOG_MAX_TEXT_CHARS,
        REPO_ROOT,
        RESULTS_ROOT,
        RESULTS_ENABLE,
        RESULTS_MODE,
        RESULTS_KEEP_LOG,
        NOVELTY_ENABLE,
        NOVELTY_INDEX_DIR,
        NOVELTY_INDEX_BUILD_BATCH_SIZE,
        NOVELTY_INDEX_BUILD_RESUME,
        NOVELTY_INDEX_BUILD_MAX_RETRIES,
        NOVELTY_INDEX_BUILD_SLEEP_SEC,
        NOVELTY_REQUIRE_EMBEDDING,
        INDEX_DIR_MODE,
        EMBEDDING_PROVIDER,
        EMBEDDING_API_URL,
    )
    from pipeline.config import PipelineConfig
    from idea2paper.infra.result_bundler import ResultBundler
    from idea2paper.infra.index_preflight import (
        validate_novelty_index,
        validate_recall_index,
        acquire_lock,
    )
    from idea2paper.infra.subdomain_taxonomy import (
        validate_subdomain_taxonomy,
        build_subdomain_taxonomy,
        resolve_subdomain_taxonomy_paths,
    )
    from idea2paper.infra.embeddings import EMBEDDING_MODEL
    from pipeline.run_logger import RunLogger
    from pipeline.run_context import set_logger, reset_logger
    from tools.build_novelty_index import build_novelty_index
    from tools.build_recall_index import build_recall_index
    from idea2paper.application.idea_packaging import IdeaPackager
except ImportError:
    # 如果直接运行脚本，尝试添加当前目录到 path
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from pipeline import Idea2StoryPipeline, OUTPUT_DIR
    from pipeline.config import (
        LOG_ROOT,
        ENABLE_RUN_LOGGING,
        LOG_MAX_TEXT_CHARS,
        REPO_ROOT,
        RESULTS_ROOT,
        RESULTS_ENABLE,
        RESULTS_MODE,
        RESULTS_KEEP_LOG,
        NOVELTY_ENABLE,
        NOVELTY_INDEX_DIR,
        NOVELTY_INDEX_BUILD_BATCH_SIZE,
        NOVELTY_INDEX_BUILD_RESUME,
        NOVELTY_INDEX_BUILD_MAX_RETRIES,
        NOVELTY_INDEX_BUILD_SLEEP_SEC,
        NOVELTY_REQUIRE_EMBEDDING,
        INDEX_DIR_MODE,
        EMBEDDING_PROVIDER,
        EMBEDDING_API_URL,
    )
    from pipeline.config import PipelineConfig
    from idea2paper.infra.result_bundler import ResultBundler
    from idea2paper.infra.index_preflight import (
        validate_novelty_index,
        validate_recall_index,
        acquire_lock,
    )
    from idea2paper.infra.subdomain_taxonomy import (
        validate_subdomain_taxonomy,
        build_subdomain_taxonomy,
        resolve_subdomain_taxonomy_paths,
    )
    from idea2paper.infra.embeddings import EMBEDDING_MODEL
    from pipeline.run_logger import RunLogger
    from pipeline.run_context import set_logger, reset_logger
    from tools.build_novelty_index import build_novelty_index
    from tools.build_recall_index import build_recall_index
    from idea2paper.application.idea_packaging import IdeaPackager


def _log_event(logger, event_type: str, payload: dict):
    if logger:
        logger.log_event(event_type, payload)


def _recall_focus_score(recall_audit: dict | None) -> float:
    if not recall_audit:
        return 0.0
    path2 = recall_audit.get("path2", {}) or {}
    candidate_stats = path2.get("candidate_stats", []) or []
    ratios = []
    for stat in candidate_stats:
        if not stat:
            continue
        before = int(stat.get("candidates_before", 0) or 0)
        after = int(stat.get("candidates_after", 0) or 0)
        if before > 0:
            ratios.append((before - after) / float(before))
    if not ratios:
        return 0.0
    return sum(ratios) / len(ratios)


def _truncate_text(text: str, max_len: int = 800) -> str:
    if not isinstance(text, str):
        return text
    return text if len(text) <= max_len else text[:max_len]


def _shrink_brief(brief: dict | None, max_len: int = 600) -> dict | None:
    if not isinstance(brief, dict):
        return None
    out = {}
    for k, v in brief.items():
        if isinstance(v, str):
            out[k] = _truncate_text(v, max_len)
        elif isinstance(v, list):
            trimmed = []
            for item in v[:5]:
                if isinstance(item, str):
                    trimmed.append(_truncate_text(item, max_len))
                else:
                    trimmed.append(item)
            out[k] = trimmed
        elif isinstance(v, dict):
            sub = {}
            for sk, sv in v.items():
                if isinstance(sv, str):
                    sub[sk] = _truncate_text(sv, max_len)
                elif isinstance(sv, list):
                    sub[sk] = [(_truncate_text(x, max_len) if isinstance(x, str) else x) for x in sv[:5]]
                else:
                    sub[sk] = sv
            out[k] = sub
        else:
            out[k] = v
    return out


def ensure_required_indexes(logger=None):
    if not PipelineConfig.INDEX_AUTO_PREPARE:
        return

    _log_event(logger, "index_preflight_start", {
        "novelty_enable": NOVELTY_ENABLE,
        "recall_use_offline_index": PipelineConfig.RECALL_USE_OFFLINE_INDEX,
        "allow_build": PipelineConfig.INDEX_ALLOW_BUILD,
        "index_dir_mode": INDEX_DIR_MODE,
        "novelty_index_dir": str(NOVELTY_INDEX_DIR),
        "recall_index_dir": str(PipelineConfig.RECALL_INDEX_DIR),
        "embedding_provider": EMBEDDING_PROVIDER,
        "embedding_api_url": EMBEDDING_API_URL,
        "embedding_model": EMBEDDING_MODEL,
    })

    # Novelty index preflight
    if NOVELTY_ENABLE:
        nodes_paper_path = OUTPUT_DIR / "nodes_paper.json"
        status = validate_novelty_index(NOVELTY_INDEX_DIR, nodes_paper_path, EMBEDDING_MODEL)
        if status.get("ok"):
            _log_event(logger, "index_preflight_ok", {"index": "novelty", "status": status})
        else:
            _log_event(logger, "index_preflight_failed", {"index": "novelty", "status": status})
            if PipelineConfig.INDEX_ALLOW_BUILD:
                lock_path = NOVELTY_INDEX_DIR / ".build.lock"
                _log_event(logger, "index_preflight_build_start", {
                    "index": "novelty",
                    "index_dir": str(NOVELTY_INDEX_DIR),
                })
                with acquire_lock(lock_path):
                    build_novelty_index(
                        index_dir=NOVELTY_INDEX_DIR,
                        batch_size=NOVELTY_INDEX_BUILD_BATCH_SIZE,
                        resume=NOVELTY_INDEX_BUILD_RESUME,
                        max_retries=NOVELTY_INDEX_BUILD_MAX_RETRIES,
                        sleep_sec=NOVELTY_INDEX_BUILD_SLEEP_SEC,
                        force_rebuild=False,
                        logger=logger,
                    )
                status = validate_novelty_index(NOVELTY_INDEX_DIR, nodes_paper_path, EMBEDDING_MODEL)
                _log_event(logger, "index_preflight_build_done", {"index": "novelty", "status": status})
                if not status.get("ok") and NOVELTY_REQUIRE_EMBEDDING:
                    raise RuntimeError("Novelty index build failed or incomplete. Please run build_novelty_index.py manually.")
            else:
                if NOVELTY_REQUIRE_EMBEDDING:
                    raise RuntimeError(
                        "Novelty index missing or mismatched. Please run: "
                        "python Paper-KG-Pipeline/scripts/tools/build_novelty_index.py --resume"
                    )
                print("⚠️ Novelty index missing/mismatch. Continuing because require_embedding=false.")

    # Recall offline index (only if enabled)
    if PipelineConfig.RECALL_USE_OFFLINE_INDEX:
        nodes_paper_path = OUTPUT_DIR / "nodes_paper.json"
        nodes_idea_path = OUTPUT_DIR / "nodes_idea.json"
        status = validate_recall_index(PipelineConfig.RECALL_INDEX_DIR, nodes_paper_path, nodes_idea_path, EMBEDDING_MODEL)
        if status.get("ok"):
            _log_event(logger, "index_preflight_ok", {"index": "recall", "status": status})
        else:
            _log_event(logger, "index_preflight_failed", {"index": "recall", "status": status})
            if PipelineConfig.INDEX_ALLOW_BUILD:
                lock_path = Path(PipelineConfig.RECALL_INDEX_DIR) / ".build.lock"
                _log_event(logger, "index_preflight_build_start", {
                    "index": "recall",
                    "index_dir": str(PipelineConfig.RECALL_INDEX_DIR),
                })
                with acquire_lock(lock_path):
                    build_recall_index(
                        index_dir=PipelineConfig.RECALL_INDEX_DIR,
                        batch_size=PipelineConfig.RECALL_EMBED_BATCH_SIZE,
                        resume=True,
                        max_retries=PipelineConfig.RECALL_EMBED_MAX_RETRIES,
                        sleep_sec=PipelineConfig.RECALL_EMBED_SLEEP_SEC,
                        force_rebuild=False,
                        logger=logger,
                    )
                status = validate_recall_index(PipelineConfig.RECALL_INDEX_DIR, nodes_paper_path, nodes_idea_path, EMBEDDING_MODEL)
                _log_event(logger, "index_preflight_build_done", {"index": "recall", "status": status})
            else:
                print("⚠️ Recall offline index missing/mismatch. Continuing with online batch fallback.")

    # Subdomain taxonomy preflight (optional)
    if PipelineConfig.SUBDOMAIN_TAXONOMY_ENABLE:
        tax_path, patterns_path = resolve_subdomain_taxonomy_paths()
        _log_event(logger, "subdomain_taxonomy_preflight_start", {
            "taxonomy_path": str(tax_path),
            "patterns_path": str(patterns_path),
            "embedding_model": EMBEDDING_MODEL,
            "embedding_api_url": EMBEDDING_API_URL,
        })
        if not patterns_path.exists():
            _log_event(logger, "subdomain_taxonomy_missing_patterns", {
                "patterns_path": str(patterns_path),
            })
            return
        status = validate_subdomain_taxonomy(tax_path, patterns_path)
        if status.get("ok"):
            _log_event(logger, "subdomain_taxonomy_preflight_ok", {"status": status})
        else:
            _log_event(logger, "subdomain_taxonomy_preflight_failed", {"status": status})
            if PipelineConfig.INDEX_ALLOW_BUILD:
                lock_path = tax_path.parent / ".subdomain_taxonomy.build.lock"
                _log_event(logger, "subdomain_taxonomy_build_start", {"taxonomy_path": str(tax_path)})
                with acquire_lock(lock_path):
                    build_subdomain_taxonomy(
                        patterns_path=patterns_path,
                        papers_path=OUTPUT_DIR / "nodes_paper.json",
                        output_path=tax_path,
                        embed_batch_size=PipelineConfig.RECALL_EMBED_BATCH_SIZE,
                        embed_max_retries=PipelineConfig.RECALL_EMBED_MAX_RETRIES,
                        embed_sleep_sec=PipelineConfig.RECALL_EMBED_SLEEP_SEC,
                        embed_timeout=120,
                        logger=logger,
                    )
                status = validate_subdomain_taxonomy(tax_path, patterns_path)
                _log_event(logger, "subdomain_taxonomy_build_done", {"status": status})
        if not status.get("ok"):
            _log_event(logger, "subdomain_taxonomy_unavailable", {"status": status})

# ===================== 主函数 =====================
def main():
    """主函数"""
    # 获取用户输入
    if len(sys.argv) > 1:
        user_idea = " ".join(sys.argv[1:])
    else:
        user_idea = "LLM-Assisted Domain Data Extraction and Cleaning"

    # 加载召回结果（调用 simple_recall_demo 的结果）
    print("📂 加载数据...")

    logger = None
    token = None
    start_time = time.time()
    start_dt = datetime.now(timezone.utc)
    run_id = f"run_{start_dt.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_{uuid.uuid4().hex[:6]}"
    success = False
    exit_code = 0

    try:
        if ENABLE_RUN_LOGGING:
            logger = RunLogger(
                base_dir=LOG_ROOT,
                run_id=run_id,
                meta={
                    "user_idea": user_idea,
                    "argv": sys.argv,
                    "entrypoint": __file__,
                },
                max_text_chars=LOG_MAX_TEXT_CHARS
            )
            token = set_logger(logger)
            logger.log_event("run_start", {"user_idea": user_idea})
            if _DOTENV_STATUS:
                logger.log_event("dotenv_loaded", _DOTENV_STATUS)
        # Preflight & auto-prepare required indexes (quality-first)
        ensure_required_indexes(logger)
        # 加载节点数据
        with open(OUTPUT_DIR / "nodes_pattern.json", 'r', encoding='utf-8') as f:
            patterns = json.load(f)
        with open(OUTPUT_DIR / "nodes_paper.json", 'r', encoding='utf-8') as f:
            papers = json.load(f)

        print(f"  ✓ 加载 {len(patterns)} 个 Pattern")
        print(f"  ✓ 加载 {len(papers)} 个 Paper")
        papers_by_id = {p.get("paper_id"): p for p in papers if p.get("paper_id")}

        # 运行召回（复用 simple_recall_demo 的逻辑）
        # 注意：这里为了复用逻辑，直接导入了 simple_recall_demo
        # 在生产环境中，建议将召回逻辑封装为独立的类

        # 临时保存原始 argv
        original_argv = sys.argv.copy()
        sys.argv = ['simple_recall_demo.py', user_idea]

        # 运行召回（使用 RecallSystem 类，支持两阶段优化）
        print("\n🔍 运行召回系统...")
        print("-" * 80)

        # 【优化】直接使用 RecallSystem 类（支持两阶段召回，大幅提速）
        from recall_system import RecallSystem

        print("  初始化召回系统...")
        recall_system = RecallSystem()

        print("\n  执行三路召回（优化版，支持两阶段加速）...")
        raw_user_idea = user_idea
        idea_brief_best = None
        retrieval_query_best = raw_user_idea
        idea_packaging_meta = None

        if PipelineConfig.IDEA_PACKAGING_ENABLE:
            try:
                packager = IdeaPackager(logger=logger)
                brief_a, query_a = packager.parse_raw_idea(raw_user_idea)
                if not query_a:
                    query_a = raw_user_idea

                first_recall = recall_system.recall(query_a, verbose=False)
                topn = max(1, int(PipelineConfig.IDEA_PACKAGING_TOPN_PATTERNS))
                candidate_k = max(1, int(PipelineConfig.IDEA_PACKAGING_CANDIDATE_K))
                top_patterns = first_recall[:topn]

                candidates = []
                judge_candidates = []
                for pattern_id, pattern_info, score in top_patterns[:candidate_k]:
                    evidence = packager.build_pattern_evidence(
                        pattern_id,
                        pattern_info,
                        papers_by_id,
                        max_exemplar_papers=PipelineConfig.IDEA_PACKAGING_MAX_EXEMPLAR_PAPERS,
                    )
                    brief_c, query_c = packager.package_with_pattern(raw_user_idea, brief_a, evidence)
                    candidates.append({
                        "pattern_id": pattern_id,
                        "pattern_name": pattern_info.get("name", ""),
                        "score": float(score),
                        "brief": brief_c,
                        "query": query_c,
                    })
                    judge_candidates.append({
                        "pattern_id": pattern_id,
                        "pattern_name": pattern_info.get("name", ""),
                        "brief": brief_c,
                    })

                best_idx, judge_info = packager.judge_best_candidate(raw_user_idea, judge_candidates)
                chosen_idx = best_idx if candidates else 0

                select_mode = (PipelineConfig.IDEA_PACKAGING_SELECT_MODE or "llm_then_recall").lower()
                recall_scores = {}
                if select_mode in ("llm_then_recall", "recall_only") and candidates:
                    for idx, cand in enumerate(candidates):
                        query = cand.get("query") or raw_user_idea
                        _ = recall_system.recall(query, verbose=False)
                        audit = getattr(recall_system, "last_audit", None)
                        recall_scores[idx] = _recall_focus_score(audit)
                    recall_best_idx = max(recall_scores, key=recall_scores.get) if recall_scores else chosen_idx
                    if select_mode == "recall_only":
                        chosen_idx = recall_best_idx
                    else:
                        if recall_scores.get(recall_best_idx, 0.0) > recall_scores.get(chosen_idx, 0.0) + 0.05:
                            chosen_idx = recall_best_idx

                chosen = candidates[chosen_idx] if candidates else None
                if chosen:
                    idea_brief_best = chosen.get("brief")
                    retrieval_query_best = chosen.get("query") or raw_user_idea
                else:
                    idea_brief_best = brief_a
                    retrieval_query_best = query_a

                idea_packaging_meta = {
                    "raw_idea": raw_user_idea,
                    "brief_a": brief_a,
                    "query_a": query_a,
                    "candidates": candidates,
                    "judge": judge_info,
                    "recall_scores": recall_scores,
                    "chosen_index": chosen_idx,
                    "query_best": retrieval_query_best,
                    "brief_best": idea_brief_best,
                }
                if logger:
                    logger.log_event("idea_packaging", {
                        "enabled": True,
                        "topn_patterns": topn,
                        "candidate_k": candidate_k,
                        "select_mode": select_mode,
                        "raw_idea": _truncate_text(raw_user_idea, 800),
                        "query_best": _truncate_text(retrieval_query_best, 800),
                        "brief_best": _shrink_brief(idea_brief_best, 600),
                        "candidates": [
                            {
                                "pattern_id": c.get("pattern_id"),
                                "pattern_name": c.get("pattern_name"),
                                "query": _truncate_text(c.get("query", ""), 300),
                            } for c in candidates
                        ],
                        "judge": judge_info,
                        "recall_scores": recall_scores,
                        "chosen_index": chosen_idx,
                    })
            except Exception as e:
                if logger:
                    logger.log_event("idea_packaging_failed", {"error": str(e)})
                idea_brief_best = None
                retrieval_query_best = raw_user_idea

        recall_results = recall_system.recall(retrieval_query_best, verbose=True)
        recall_audit = getattr(recall_system, "last_audit", None)

        # 【关键修复】加载完整的 patterns_structured.json 以合并数据
        patterns_structured_file = OUTPUT_DIR / "patterns_structured.json"
        if patterns_structured_file.exists():
            with open(patterns_structured_file, 'r', encoding='utf-8') as f:
                patterns_structured = json.load(f)

            # 构建 pattern_id -> structured_data 的映射
            structured_map = {}
            for p in patterns_structured:
                pattern_id = f"pattern_{p.get('pattern_id')}"
                structured_map[pattern_id] = p

            # 合并 skeleton_examples 和 common_tricks 到召回结果
            merged_results = []
            for pattern_id, pattern_info, score in recall_results:
                merged_pattern = dict(pattern_info)
                if pattern_id in structured_map:
                    merged_pattern['skeleton_examples'] = structured_map[pattern_id].get('skeleton_examples', [])
                    merged_pattern['common_tricks'] = structured_map[pattern_id].get('common_tricks', [])
                merged_results.append((pattern_id, merged_pattern, score))

            recalled_patterns = merged_results
        else:
            # 如果没有 patterns_structured.json，直接使用召回结果
            recalled_patterns = recall_results

        # 加载 papers 数据 (Pipeline 需要用于 RAG 查重)
        print("\n  加载 Papers 数据用于查重...")
        with open(OUTPUT_DIR / "nodes_paper.json", 'r', encoding='utf-8') as f:
            papers = json.load(f)

        # 恢复 argv
        sys.argv = original_argv

        print("-" * 80)
        print(f"✅ 召回完成: Top-{len(recalled_patterns)} Patterns\n")

        if not recalled_patterns:
            raise RuntimeError(
                "Recall returned 0 patterns. Please provide a more detailed idea "
                "(preferably an English sentence with domain/method keywords)."
            )

        # 运行 Pipeline（传递 user_idea 用于 Pattern 智能分类）
        pipeline = Idea2StoryPipeline(
            raw_user_idea,
            recalled_patterns,
            papers,
            run_id=run_id,
            idea_brief=idea_brief_best,
        )
        result = pipeline.run()
        if not isinstance(result, dict):
            raise RuntimeError(f"Pipeline returned invalid result type: {type(result).__name__}")
        if not result.get("success", False):
            raise RuntimeError(
                "Pipeline did not produce a valid story. "
                "Try a more specific idea and ensure recall can retrieve patterns."
            )
        if "final_story" not in result:
            raise RuntimeError("Pipeline result missing required field: final_story")

        if recall_audit is not None:
            result["recall_audit"] = recall_audit
            if logger and PipelineConfig.RECALL_AUDIT_IN_EVENTS:
                logger.log_event("recall_audit", recall_audit)
        if idea_packaging_meta:
            result["idea_packaging"] = idea_packaging_meta
        success = True

        # 保存结果
        output_file = OUTPUT_DIR / "final_story.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result['final_story'], f, ensure_ascii=False, indent=2)

        print(f"\n💾 最终 Story 已保存到: {output_file}")

        # 保存完整结果
        full_result_file = OUTPUT_DIR / "pipeline_result.json"
        results_dir = str(RESULTS_ROOT / run_id) if RESULTS_ENABLE else None
        with open(full_result_file, 'w', encoding='utf-8') as f:
            json.dump({
                'user_idea': user_idea,
                'success': result['success'],
                'iterations': result['iterations'],
                'selected_patterns': result['selected_patterns'],
                'final_story': result['final_story'],
                'review_history': result['review_history'],
                'results_dir': results_dir,
                'novelty_report': result.get('novelty_report'),
                'recall_audit': result.get('recall_audit'),
                'review_summary': {
                    'total_reviews': len(result['review_history']),
                    'final_score': result['review_history'][-1]['avg_score'] if result['review_history'] else 0
                },
                'refinement_summary': {
                    'total_refinements': len(result['refinement_history']),
                    'issues_addressed': [r['issue'] for r in result['refinement_history']]
                },
                'verification_summary': {
                    'collision_detected': result['verification_result']['collision_detected'],
                    'max_similarity': result['verification_result']['max_similarity']
                },
                'idea_packaging': result.get('idea_packaging')
            }, f, ensure_ascii=False, indent=2)

        print(f"💾 完整结果已保存到: {full_result_file}")

        # 聚合产物到 repo 根 results/
        if RESULTS_ENABLE:
            try:
                bundler = ResultBundler(
                    repo_root=REPO_ROOT,
                    results_root=RESULTS_ROOT,
                    mode=RESULTS_MODE,
                    keep_log=RESULTS_KEEP_LOG,
                )
                run_log_dir = (LOG_ROOT / run_id) if ENABLE_RUN_LOGGING else None
                novelty_report_path = None
                if isinstance(result.get("novelty_report"), dict):
                    novelty_report_path = result["novelty_report"].get("report_path")
                bundle_status = bundler.bundle(
                    run_id=run_id,
                    user_idea=user_idea,
                    success=success,
                    output_dir=OUTPUT_DIR,
                    run_log_dir=run_log_dir,
                    extra={
                        "config_snapshot": {
                            "results": {
                                "enable": RESULTS_ENABLE,
                                "dir": str(RESULTS_ROOT),
                                "mode": RESULTS_MODE,
                                "keep_log": RESULTS_KEEP_LOG,
                            },
                            "logging": {
                                "enable": ENABLE_RUN_LOGGING,
                                "dir": str(LOG_ROOT),
                                "max_text_chars": LOG_MAX_TEXT_CHARS,
                            },
                            "critic": {
                                "strict_json": PipelineConfig.CRITIC_STRICT_JSON,
                                "json_retries": PipelineConfig.CRITIC_JSON_RETRIES,
                            },
                            "pass": {
                                "mode": PipelineConfig.PASS_MODE,
                                "min_pattern_papers": PipelineConfig.PASS_MIN_PATTERN_PAPERS,
                                "fallback": PipelineConfig.PASS_FALLBACK,
                                "fixed_score": PipelineConfig.PASS_SCORE,
                            },
                        },
                        "novelty_report_path": novelty_report_path
                    },
                )
                if bundle_status.get("ok"):
                    print(f"✅ Results bundled to: {bundle_status.get('results_dir')}")
                    if logger:
                        logger.log_event("results_bundled", {
                            "results_dir": bundle_status.get("results_dir"),
                            "mode": RESULTS_MODE,
                            "partial": bundle_status.get("partial", False)
                        })
                else:
                    if logger:
                        logger.log_event("results_bundle_failed", {
                            "errors": bundle_status.get("errors", []),
                            "mode": RESULTS_MODE
                        })
            except Exception as e:
                print(f"[results] warning: bundling failed: {e}")
                if logger:
                    logger.log_event("results_bundle_failed", {"error": str(e)})

    except Exception as e:
        success = False
        exit_code = 1
        print(f"\n❌ 错误: {e}")
        if logger:
            logger.log_event("run_error", {"error": str(e)})
        import traceback
        traceback.print_exc()
    finally:
        if logger:
            logger.log_event("run_end", {
                "success": success,
                "duration_ms": int((time.time() - start_time) * 1000)
            })
        if token is not None:
            reset_logger(token)

    if exit_code != 0:
        raise SystemExit(exit_code)


if __name__ == '__main__':
    main()
