# 中文注释: 本文件实现状态转移动作。
from __future__ import annotations

import copy
from typing import Optional

from paper_kg.interfaces import IGraphStore
from paper_kg.models import Defect, Pattern, Trick
from paper_kg.story_state import StoryState


class Action:
    """
    功能：故事状态转移动作集合。
    参数：无。
    返回：新的 StoryState。
    流程：基于 deepcopy 执行无副作用更新。
    说明：避免直接修改原状态。
    """

    @staticmethod
    def add_pattern(
        state: StoryState, pattern: Pattern, tricks: Optional[list[Trick]] = None
    ) -> StoryState:
        """
        功能：添加一个新的故事套路。
        参数：state、pattern、tricks。
        返回：新的 StoryState。
        流程：复制状态 → 添加套路与技巧。
        说明：tricks 为空时仅添加套路，不自动推导技巧。
        """
        new_state = copy.deepcopy(state)
        new_state.patterns.append(pattern)
        if tricks:
            new_state.tricks.extend(tricks)
        return new_state

    @staticmethod
    def merge_patterns(
        state: StoryState,
        pattern1: Pattern,
        pattern2: Pattern,
        graph: IGraphStore,
    ) -> StoryState:
        """
        功能：合并两个套路。
        参数：state、pattern1、pattern2、graph。
        返回：新的 StoryState。
        流程：查询合并结果 → 替换原套路。
        说明：找不到合并结果时保留原套路。
        """
        new_state = copy.deepcopy(state)
        merged = graph.find_merged_pattern(pattern1.id, pattern2.id)
        if merged:
            new_state.patterns = [
                p
                for p in new_state.patterns
                if p.id not in {pattern1.id, pattern2.id}
            ]
            new_state.patterns.append(merged)
        return new_state

    @staticmethod
    def fix_defect(state: StoryState, defect: Defect, trick: Trick) -> StoryState:
        """
        功能：使用 Trick 修复缺陷。
        参数：state、defect、trick。
        返回：新的 StoryState。
        流程：复制状态 → 移除缺陷 → 添加 Trick。
        说明：若缺陷不存在则仅追加 Trick。
        """
        new_state = copy.deepcopy(state)
        new_state.defects = [d for d in new_state.defects if d.defect_id != defect.defect_id]
        new_state.tricks.append(trick)
        return new_state

    @staticmethod
    def extend_section(
        state: StoryState, section: str, pattern: Pattern
    ) -> StoryState:
        """
        功能：强化某个章节的套路。
        参数：state、section、pattern。
        返回：新的 StoryState。
        流程：复制状态 → 追加套路。
        说明：section 仅用于日志语义。
        """
        new_state = copy.deepcopy(state)
        new_state.patterns.append(pattern)
        return new_state
