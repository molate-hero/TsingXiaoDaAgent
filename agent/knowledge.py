"""知识库：加载 minors/ 下的辅修培养方案 Markdown，提供目录、检索、详情能力。

目录结构约定：
    minors/
        清华大学本科生辅修学士学位专业教学管理办法.md   → 教务处管理办法
        overview.md                                    → 专业总目录
        <院系>/
            <专业>.md                                  → 各专业培养方案
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

REGULATION_KEY = "辅修学士学位管理办法"
OVERVIEW_KEY = "辅修专业目录"
DEFAULT_MAX_DETAIL_CHARS = 6000


@dataclass
class MinorDoc:
    key: str            # 唯一键（专业名 / 管理办法 / 目录）
    department: str     # 院系（文件所在目录名）
    title: str          # 文档标题（首个 Markdown 标题或文件名）
    path: Path
    content: str


class KnowledgeBase:
    def __init__(self, root: Path, max_detail_chars: int = DEFAULT_MAX_DETAIL_CHARS):
        self.root = root
        self.max_detail_chars = max_detail_chars
        self.docs: list[MinorDoc] = []
        self.by_key: dict[str, MinorDoc] = {}

    def load(self) -> "KnowledgeBase":
        """递归加载知识库目录下所有 *.md。"""
        self.docs = []
        self.by_key = {}
        if not self.root.is_dir():
            raise FileNotFoundError(f"知识库目录不存在: {self.root}")
        for path in sorted(self.root.rglob("*.md")):
            rel = path.relative_to(self.root)
            content = path.read_text(encoding="utf-8")
            heading = re.search(r"^#{1,3}\s+(.+)$", content, re.M)
            title = heading.group(1).strip() if heading else path.stem
            if path.name == "overview.md":
                key, department = OVERVIEW_KEY, "总目录"
            elif "管理办法" in path.name:
                key, department = REGULATION_KEY, "教务处"
            else:
                key = path.stem
                department = rel.parts[0] if len(rel.parts) > 1 else "未分类"
            doc = MinorDoc(key=key, department=department, title=title, path=path, content=content)
            self.docs.append(doc)
            self.by_key[key] = doc
        return self

    def list_minors(self) -> list[dict]:
        """按院系分组的专业目录（排除总目录与管理办法）。"""
        groups: dict[str, list[str]] = {}
        for doc in self.docs:
            if doc.key in (REGULATION_KEY, OVERVIEW_KEY):
                continue
            groups.setdefault(doc.department, []).append(doc.key)
        return [
            {"department": dept, "minors": sorted(names)}
            for dept, names in sorted(groups.items())
        ]

    def get_minor(self, name: str) -> MinorDoc | None:
        """按专业名取文档：精确 → 包含（双向）→ 长度最接近者。"""
        target = (name or "").strip().rstrip("辅修")
        if not target:
            return None
        if target in self.by_key:
            return self.by_key[target]
        hits = [d for d in self.docs if target in d.key or d.key in target]
        if hits:
            return min(hits, key=lambda d: abs(len(d.key) - len(target)))
        return None

    def get_regulations(self) -> MinorDoc | None:
        return self.by_key.get(REGULATION_KEY)

    def get_detail(self, name: str, max_chars: int | None = None) -> str:
        """返回供工具使用的详情文本（超长截断）。"""
        doc = self.get_minor(name)
        if doc is None:
            available = "、".join(sorted(self.by_key))
            return f"知识库中未收录「{name}」。已收录：{available}"
        limit = max_chars or self.max_detail_chars
        content = doc.content
        if len(content) > limit:
            content = content[:limit] + f"\n\n……（内容过长，已截断，全文共 {len(doc.content)} 字符）"
        return f"【{doc.title}】（{doc.department}）\n{content}"

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """关键词检索：按词频打分（命中专业名加权），返回前 top_k 条含摘要。"""
        tokens = [t for t in re.split(r"[\s,，、;；]+", query or "") if t]
        if not tokens:
            return []
        scored: list[tuple[float, MinorDoc]] = []
        for doc in self.docs:
            score = 0.0
            for tok in tokens:
                n = doc.content.count(tok)
                if n:
                    score += n * (2.0 if tok in doc.key else 1.0)
            if score > 0:
                scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "name": doc.key,
                "department": doc.department,
                "score": round(score, 2),
                "snippet": self._snippet(doc, tokens),
            }
            for score, doc in scored[:top_k]
        ]

    @staticmethod
    def _snippet(doc: MinorDoc, tokens: list[str], radius: int = 160) -> str:
        """取首个命中词附近 ±radius 字符作为摘要。"""
        pos = -1
        for tok in tokens:
            p = doc.content.find(tok)
            if p >= 0 and (pos < 0 or p < pos):
                pos = p
        if pos < 0:
            return doc.content[: radius * 2].replace("\n", " ")
        start, end = max(0, pos - radius), min(len(doc.content), pos + radius)
        return (
            ("…" if start else "")
            + doc.content[start:end].replace("\n", " ")
            + ("…" if end < len(doc.content) else "")
        )
