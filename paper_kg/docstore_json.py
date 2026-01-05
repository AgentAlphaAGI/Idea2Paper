# 中文注释: 本文件提供基于 JSON 文件的原始文档存储实现。
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from paper_kg.interfaces import IDocStore


class JsonDocStore(IDocStore):
    """
    功能：本地 JSON 文档存储。
    参数：path。
    返回：JsonDocStore 实例。
    流程：读取/写入 JSON 文件。
    说明：用于原型阶段保存原始论文/审稿数据。
    """

    def __init__(self, path: str) -> None:
        """
        功能：初始化存储路径并加载数据。
        参数：path。
        返回：无。
        流程：创建路径 → 读取 JSON。
        说明：文件不存在时初始化空结构。
        """
        self.path = Path(path)
        self.data = self._load()

    def _load(self) -> Dict:
        """
        功能：加载 JSON 文件。
        参数：无。
        返回：数据字典。
        流程：读取文件 → 解析 JSON。
        说明：不存在时返回空结构。
        """
        if not self.path.exists():
            return {"papers": {}, "reviews": {}}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _save(self) -> None:
        """
        功能：保存 JSON 文件。
        参数：无。
        返回：无。
        流程：序列化 → 写入磁盘。
        说明：覆盖旧文件。
        """
        self.path.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def save_raw_paper(self, paper_id: str, raw: Dict) -> None:
        """
        功能：保存原始论文数据。
        参数：paper_id、raw。
        返回：无。
        流程：写入内存字典 → 保存文件。
        说明：重复 ID 会覆盖。
        """
        self.data.setdefault("papers", {})[paper_id] = raw
        self._save()

    def save_raw_review(self, review_id: str, raw: Dict) -> None:
        """
        功能：保存原始审稿数据。
        参数：review_id、raw。
        返回：无。
        流程：写入内存字典 → 保存文件。
        说明：重复 ID 会覆盖。
        """
        self.data.setdefault("reviews", {})[review_id] = raw
        self._save()

    def load_raw_paper(self, paper_id: str) -> Optional[Dict]:
        """
        功能：读取原始论文数据。
        参数：paper_id。
        返回：原始字典或 None。
        流程：读取内存字典。
        说明：不存在则返回 None。
        """
        return self.data.get("papers", {}).get(paper_id)

    def load_raw_review(self, review_id: str) -> Optional[Dict]:
        """
        功能：读取原始审稿数据。
        参数：review_id。
        返回：原始字典或 None。
        流程：读取内存字典。
        说明：不存在则返回 None。
        """
        return self.data.get("reviews", {}).get(review_id)
