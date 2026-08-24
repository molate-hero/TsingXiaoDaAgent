"""课程库：加载 minors/curated_courses.json，提供课程检索、详情、按辅修方案列出课程。

数据源结构（每门课程一个对象）：
- id / name / name_en / department / credits / total_hours / hours_lecture / hours_lab / hours_extra
- course_type / classification / language / target_major / grade_level / prerequisites
- description / description_en / objectives / expected_outcomes
- assessment_method / assessment_detail / grade_breakdown / textbooks / instructor / team
- search_text                 （已拼接的可检索文本）
- minor_programs              （该课程出现在哪些辅修培养方案中）
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

DEFAULT_MAX_DETAIL_CHARS = 4000

# 详情展示需要的字段
_KEY_FIELDS = (
    "id",
    "name",
    "department",
    "credits",
    "total_hours",
    "hours_lecture",
    "hours_lab",
    "course_type",
    "language",
    "prerequisites",
    "description",
    "objectives",
    "expected_outcomes",
    "assessment_method",
    "assessment_detail",
    "grade_breakdown",
    "textbooks",
    "instructor",
)


class CourseIndex:
    def __init__(self, path: Path, max_detail_chars: int = DEFAULT_MAX_DETAIL_CHARS):
        self.path = path
        self.max_detail_chars = max_detail_chars
        self.courses: list[dict] = []
        self.by_id: dict[str, dict] = {}
        self.by_name: dict[str, list[dict]] = {}
        self.programs: dict[str, list[dict]] = {}  # 培养方案名 -> 课程列表
        self.loaded = False

    def load(self) -> "CourseIndex":
        """加载课程库 JSON；文件缺失/损坏时降级为空库（不阻塞启动）。"""
        self.courses, self.by_id, self.by_name, self.programs = [], {}, {}, {}
        if not self.path.is_file():
            return self
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return self
        for item in data:
            if not isinstance(item, dict) or "id" not in item:
                continue
            self.courses.append(item)
            self.by_id[item["id"]] = item
            self.by_name.setdefault(item.get("name", ""), []).append(item)
            for mp in item.get("minor_programs", []):
                program = mp.get("program", "")
                if program:
                    self.programs.setdefault(program, []).append(item)
        self.loaded = True
        return self

    def __len__(self) -> int:
        return len(self.courses)

    # ---- 查询 ----
    def find(self, course: str) -> list[dict]:
        """按课程号或课程名定位，返回可能的多条（存在重名课程时）。"""
        target = (course or "").strip()
        if not target:
            return []
        if target in self.by_id:
            return [self.by_id[target]]
        if target in self.by_name:
            return self.by_name[target]  # 可能同名多门
        hits = [c for c in self.courses if target in c.get("name", "")]
        if hits:
            return [min(hits, key=lambda c: abs(len(c.get("name", "")) - len(target)))]
        return []

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """关键词检索：按词频打分（命中课程名加权），返回排名靠前的课程概要。"""
        tokens = [t for t in re.split(r"[\s,，、;；/]+", query or "") if t]
        if not tokens:
            return []
        scored: list[tuple[float, dict]] = []
        for c in self.courses:
            name = c.get("name", "")
            score = 0.0
            for tok in tokens:
                n = (c.get("search_text", "") or "").count(tok) + c.get("name_en", "").count(tok)
                if n:
                    score += n * (3.0 if tok in name else 1.0)
            if score > 0:
                scored.append((score, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:top_k]]

    def by_program(self, program: str, fuzzy: bool = True) -> tuple[list[dict], str | None]:
        """按辅修培养方案列出课程。返回 (课程列表, 匹配到的方案名)；未匹配时列表为空。"""
        target = (program or "").strip()
        if not target:
            return [], None
        if target in self.programs:
            return self.programs[target], target
        if fuzzy:
            # 去掉"专业辅修培养方案"等后缀做包含匹配
            candidates = [p for p in self.programs if target in p or p in target]
            if candidates:
                matched = min(candidates, key=lambda p: abs(len(p) - len(target)))
                return self.programs[matched], matched
        return [], None

    # ---- 格式化 ----
    def summary(self, c: dict) -> str:
        progs = [mp.get("program", "") for mp in c.get("minor_programs", [])]
        return "\n".join(
            [
                f"· {c.get('name', '')}（{c.get('department', '')}）课程号 {c.get('id', '')}，"
                f"{c.get('credits', '')} 学分，先修：{c.get('prerequisites', '') or '无'}",
                f"  简介：{(c.get('description', '') or '')[:60]}",
                f"  所属辅修：{'、'.join(progs) if progs else '—'}",
            ]
        )

    def detail(self, c: dict, max_chars: int | None = None) -> str:
        limit = max_chars or self.max_detail_chars
        lines = [
            f"【{c.get('name', '')}】（{c.get('department', '')}）",
            f"课程号：{c.get('id', '')}",
            f"学分：{c.get('credits', '')} | 学时：{c.get('total_hours', '')}"
            f"（讲授 {c.get('hours_lecture', '')} / 实验 {c.get('hours_lab', '')}）",
            f"课程类型：{c.get('course_type', '')} | 语言：{c.get('language', '')}",
            f"先修要求：{c.get('prerequisites', '') or '无'}",
        ]
        for key, label in (
            ("description", "课程简介"),
            ("objectives", "教学目标"),
            ("expected_outcomes", "预期成果"),
            ("assessment_method", "考核方式"),
            ("grade_breakdown", "成绩构成"),
            ("textbooks", "教材"),
            ("instructor", "教师"),
        ):
            value = str(c.get(key, "") or "").strip()
            if value:
                lines.append(f"{label}：{value}")
        progs = [mp.get("program", "") for mp in c.get("minor_programs", [])]
        if progs:
            lines.append("所属辅修培养方案：" + "、".join(progs))
        text = "\n".join(lines)
        if len(text) > limit:
            text = text[:limit] + f"\n……（内容过长，已截断，全文共 {len(text)} 字符）"
        return text
