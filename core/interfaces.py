# 中文注释: 本文件包含可运行代码与流程说明。
from __future__ import annotations

from typing import Iterable, List, Optional, Protocol

from core.models import AdCase, AdDraft, CreativePattern, IntentResult, RetrievedAd


class ICreativeLibrary(Protocol):
    def list_patterns(self) -> List[CreativePattern]:
        """
        功能：list_patterns 的核心流程封装，负责处理输入并输出结果。
        参数：无。
        返回：依据实现返回结果或产生副作用。
        流程：接收输入 → 核心逻辑处理 → 输出或触发副作用。
        说明：如遇异常或边界情况，调用侧需关注返回值或日志信息。
        """
        ...

    def get_random_pattern(
        self,
        rng,
        industry_id: str,
        exclude_ids: Optional[set[str]] = None,
    ) -> CreativePattern:
        """
        功能：get_random_pattern 的核心流程封装，负责处理输入并输出结果。
        参数：rng、industry_id、exclude_ids。
        返回：依据实现返回结果或产生副作用。
        流程：接收输入 → 核心逻辑处理 → 输出或触发副作用。
        说明：如遇异常或边界情况，调用侧需关注返回值或日志信息。
        """
        ...


class IAdRepository(Protocol):
    def list_ads(self) -> List[AdCase]:
        """
        功能：list_ads 的核心流程封装，负责处理输入并输出结果。
        参数：无。
        返回：依据实现返回结果或产生副作用。
        流程：接收输入 → 核心逻辑处理 → 输出或触发副作用。
        说明：如遇异常或边界情况，调用侧需关注返回值或日志信息。
        """
        ...

    def filter_by_industry(self, industry_id: str) -> List[AdCase]:
        """
        功能：filter_by_industry 的核心流程封装，负责处理输入并输出结果。
        参数：industry_id。
        返回：依据实现返回结果或产生副作用。
        流程：接收输入 → 核心逻辑处理 → 输出或触发副作用。
        说明：如遇异常或边界情况，调用侧需关注返回值或日志信息。
        """
        ...


class IEmbeddingModel(Protocol):
    def embed(self, text: str) -> List[float]:
        """
        功能：embed 的核心流程封装，负责处理输入并输出结果。
        参数：text。
        返回：依据实现返回结果或产生副作用。
        流程：接收输入 → 核心逻辑处理 → 输出或触发副作用。
        说明：如遇异常或边界情况，调用侧需关注返回值或日志信息。
        """
        ...


class IReranker(Protocol):
    def score(self, query: str, docs: List[str]) -> List[float]:
        """
        功能：score 的核心流程封装，负责处理输入并输出结果。
        参数：query、docs。
        返回：依据实现返回结果或产生副作用。
        流程：接收输入 → 核心逻辑处理 → 输出或触发副作用。
        说明：如遇异常或边界情况，调用侧需关注返回值或日志信息。
        """
        ...


class IRetriever(Protocol):
    def retrieve(self, query: str, industry_id: str, top_k: int) -> List[RetrievedAd]:
        """
        功能：retrieve 的核心流程封装，负责处理输入并输出结果。
        参数：query、industry_id、top_k。
        返回：依据实现返回结果或产生副作用。
        流程：接收输入 → 核心逻辑处理 → 输出或触发副作用。
        说明：如遇异常或边界情况，调用侧需关注返回值或日志信息。
        """
        ...


class ILLMProvider(Protocol):
    def generate(
        self, intent: IntentResult, pattern: CreativePattern, attempt: int
    ) -> AdDraft:
        """
        功能：generate 的核心流程封装，负责处理输入并输出结果。
        参数：intent、pattern、attempt。
        返回：依据实现返回结果或产生副作用。
        流程：接收输入 → 核心逻辑处理 → 输出或触发副作用。
        说明：如遇异常或边界情况，调用侧需关注返回值或日志信息。
        """
        ...


class IReviewerGroup(Protocol):
    def review(
        self, ad_draft: AdDraft, intent: IntentResult, pattern: CreativePattern
    ) -> List:
        """
        功能：review 的核心流程封装，负责处理输入并输出结果。
        参数：ad_draft、intent、pattern。
        返回：依据实现返回结果或产生副作用。
        流程：接收输入 → 核心逻辑处理 → 输出或触发副作用。
        说明：如遇异常或边界情况，调用侧需关注返回值或日志信息。
        """
        ...
