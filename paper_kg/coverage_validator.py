# Section coverage validator (semantic checks + evidence).
from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional, Protocol

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from core.config import PaperWorkflowConfig
from core.llm_utils import coerce_model, extract_json, log_llm_output, safe_content
from paper_kg.models import PaperSectionOutput, PaperSectionPlan, RequiredPoint
from paper_kg.prompts import load_paper_prompt


logger = logging.getLogger(__name__)


class CoverageCheckResult(BaseModel):
    """
    Coverage check result.
    - ok: overall pass/fail.
    - missing_point_ids / missing_trick_names: unmet requirements.
    - evidence: snippets copied from the section text.
    """

    model_config = {
        "extra": "ignore",
    }

    ok: bool = False
    missing_point_ids: List[str] = Field(default_factory=list)
    missing_trick_names: List[str] = Field(default_factory=list)
    evidence: Dict[str, str] = Field(default_factory=dict)
    notes: str = ""


class ISectionCoverageValidator(Protocol):
    def check(
        self, section_plan: PaperSectionPlan, section_output: PaperSectionOutput
    ) -> CoverageCheckResult:
        """Check whether the section covers required points/tricks."""


class LLMSectionCoverageValidator:
    """LLM-based semantic coverage validator (quality-first)."""

    def __init__(
        self,
        llm: BaseChatModel,
        require_evidence: bool = True,
        log_outputs: bool = False,
        max_chars: int = 4000,
    ) -> None:
        self.llm = llm
        self.require_evidence = require_evidence
        self.log_outputs = log_outputs
        self.max_chars = max_chars
        self.structured_llm = llm.with_structured_output(CoverageCheckResult, method="json_mode")
        system_prompt = load_paper_prompt("coverage_check_system.txt")
        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_prompt),
                (
                    "human",
                    "section_plan={section_plan}\n"
                    "required_point_items={required_point_items}\n"
                    "required_trick_names={required_trick_names}\n"
                    "content_markdown=\n{content_markdown}",
                ),
            ]
        )

    def check(
        self, section_plan: PaperSectionPlan, section_output: PaperSectionOutput
    ) -> CoverageCheckResult:
        required_items = _resolve_required_point_items(section_plan)
        payload = {
            "section_plan": json.dumps(section_plan.model_dump(), ensure_ascii=False),
            "required_point_items": json.dumps(
                [item.model_dump() for item in required_items], ensure_ascii=False
            ),
            "required_trick_names": json.dumps(
                section_plan.required_trick_names or _extract_trick_names(section_plan.required_tricks),
                ensure_ascii=False,
            ),
            "content_markdown": section_output.content_markdown or "",
        }
        try:
            response = self.structured_llm.invoke(self.prompt.format_messages(**payload))
            if self.log_outputs:
                log_llm_output(logger, "COVERAGE_CHECK", response, self.max_chars)
            result = coerce_model(response, CoverageCheckResult)
        except Exception as exc:  # noqa: BLE001
            logger.warning("COVERAGE_VALIDATE_FAIL: %s", exc)
            raw = self.llm.invoke(self.prompt.format_messages(**payload))
            if self.log_outputs:
                log_llm_output(logger, "COVERAGE_CHECK_RAW", raw, self.max_chars)
            data = extract_json(safe_content(raw))
            result = CoverageCheckResult.model_validate(data)
        return _enforce_evidence(result, required_items, section_plan, self.require_evidence)


class RuleCoverageValidator:
    """Rule-based coverage validator (offline/weak check)."""

    def check(
        self, section_plan: PaperSectionPlan, section_output: PaperSectionOutput
    ) -> CoverageCheckResult:
        required_items = _resolve_required_point_items(section_plan)
        required_tricks = section_plan.required_trick_names or _extract_trick_names(
            section_plan.required_tricks
        )
        content = section_output.content_markdown or ""
        covered_points = set(section_output.covered_points or [])
        used_tricks = set(section_output.used_tricks or [])

        missing_point_ids: List[str] = []
        evidence: Dict[str, str] = {}
        for item in required_items:
            if item.id in covered_points or item.text in covered_points:
                evidence[item.id] = item.text or "covered"
                continue
            if item.text and item.text in content:
                evidence[item.id] = item.text
                continue
            missing_point_ids.append(item.id)

        missing_trick_names: List[str] = []
        for name in required_tricks:
            if name in used_tricks:
                evidence[name] = name
                continue
            if name and name in content:
                evidence[name] = name
                continue
            missing_trick_names.append(name)

        ok = not missing_point_ids and not missing_trick_names
        return CoverageCheckResult(
            ok=ok,
            missing_point_ids=missing_point_ids,
            missing_trick_names=missing_trick_names,
            evidence=evidence,
            notes="rule",
        )


def build_coverage_validator(
    llm: Optional[BaseChatModel], config: PaperWorkflowConfig
) -> Optional[ISectionCoverageValidator]:
    backend = (config.coverage_validator_backend or "off").lower()
    if backend == "off":
        return None
    if backend == "rule":
        return RuleCoverageValidator()
    if backend == "llm":
        if llm is None:
            if config.quality_fail_fast:
                return None
            return RuleCoverageValidator()
        return LLMSectionCoverageValidator(
            llm=llm,
            require_evidence=bool(config.coverage_validator_require_evidence),
            log_outputs=bool(config.log_llm_outputs),
            max_chars=int(config.log_llm_output_max_chars),
        )
    return None


def _resolve_required_point_items(section_plan: PaperSectionPlan) -> List[RequiredPoint]:
    if section_plan.required_point_items:
        return list(section_plan.required_point_items)
    items: List[RequiredPoint] = []
    for idx, text in enumerate(section_plan.required_points or [], start=1):
        items.append(RequiredPoint(id=f"{section_plan.section_id}_p{idx}", text=text))
    return items


def _extract_trick_names(required_tricks: List[str]) -> List[str]:
    names: List[str] = []
    for instruction in required_tricks or []:
        if not instruction:
            continue
        if "\\uFF1A" in instruction:
            name = instruction.split("\\uFF1A", 1)[0].strip()
        elif ":" in instruction:
            name = instruction.split(":", 1)[0].strip()
        else:
            name = instruction.strip()
        if name:
            names.append(name)
    return names


def _enforce_evidence(
    result: CoverageCheckResult,
    required_items: List[RequiredPoint],
    section_plan: PaperSectionPlan,
    require_evidence: bool,
) -> CoverageCheckResult:
    missing_point_ids = set(result.missing_point_ids or [])
    missing_trick_names = set(result.missing_trick_names or [])
    if require_evidence:
        covered_ids = {item.id for item in required_items} - missing_point_ids
        for point_id in covered_ids:
            evidence = (result.evidence or {}).get(point_id, "").strip()
            if not evidence:
                missing_point_ids.add(point_id)
        required_tricks = section_plan.required_trick_names or _extract_trick_names(
            section_plan.required_tricks
        )
        covered_tricks = set(required_tricks) - missing_trick_names
        for trick_name in covered_tricks:
            evidence = (result.evidence or {}).get(trick_name, "").strip()
            if not evidence:
                missing_trick_names.add(trick_name)

    ok = not missing_point_ids and not missing_trick_names
    return result.model_copy(
        update={
            "ok": ok,
            "missing_point_ids": sorted(missing_point_ids),
            "missing_trick_names": sorted(missing_trick_names),
        }
    )
