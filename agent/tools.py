"""ReAct 工具集：基于知识库的辅修信息检索工具 + 课程查询工具。

每个 Tool 声明：名称、用途说明、JSON Schema 参数（供模型生成 Action Input）、异步实现。
新增工具只需在 build_tools 中追加一个 Tool 即可。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from .courses import CourseIndex
from .knowledge import KnowledgeBase


@dataclass
class Tool:
    name: str
    description: str
    func: Callable[..., Awaitable[str]]
    parameters: dict[str, Any] = field(default_factory=dict)  # JSON Schema（type/properties/required）

    def schema_text(self) -> str:
        if not self.parameters:
            return "（无参数）"
        return json.dumps(self.parameters, ensure_ascii=False)


async def _list_minors(kb: KnowledgeBase) -> str:
    groups = kb.list_minors()
    lines = ["清华大学本科辅修专业目录（按院系分组）："]
    for g in groups:
        lines.append(f"- {g['department']}：{'、'.join(g['minors'])}")
    lines.append("提示：需要某专业详情请用 get_minor_detail；需要按关键词定位请用 search_minors。")
    return "\n".join(lines)


async def _get_minor_detail(kb: KnowledgeBase, name: str) -> str:
    return kb.get_detail(name)


async def _search_minors(kb: KnowledgeBase, query: str) -> str:
    results = kb.search(query, top_k=3)
    if not results:
        return f"未检索到与「{query}」相关的内容。"
    lines = [f"检索「{query}」结果（前 {len(results)} 条）："]
    for r in results:
        lines.append(
            f"- {r['name']}（{r['department']}，相关度 {r['score']}）\n  摘要：{r['snippet']}"
        )
    lines.append("需要完整培养方案请用 get_minor_detail 获取。")
    return "\n".join(lines)


async def _get_regulations(kb: KnowledgeBase) -> str:
    doc = kb.get_regulations()
    if doc is None:
        return "知识库未收录《清华大学本科生辅修学士学位专业教学管理办法》。"
    return f"【{doc.title}】\n{doc.content[: kb.max_detail_chars]}"


async def _search_courses(cidx: CourseIndex, query: str) -> str:
    results = cidx.search(query, top_k=5)
    if not results:
        return f"课程库中未检索到与「{query}」相关的课程。可换个关键词，或先查询辅修培养方案中的课程。"
    lines = [f"检索「{query}」课程结果（前 {len(results)} 条）："]
    for c in results:
        lines.append(cidx.summary(c))
    lines.append("需要某门课详情请用 get_course_detail。")
    return "\n".join(lines)


async def _get_course_detail(cidx: CourseIndex, course: str) -> str:
    matches = cidx.find(course)
    if not matches:
        return f"课程库中未找到「{course}」。可用 search_courses 检索，或查看某辅修培养方案中的课程。"
    if len(matches) > 1:
        # 重名课程：列出候选，让模型用课程号明确目标
        lines = [f"课程名「{course}」存在多门，请用课程号明确后再查："]
        for c in matches:
            lines.append(cidx.summary(c))
        return "\n".join(lines)
    return cidx.detail(matches[0])


async def _list_courses_for_minor(cidx: CourseIndex, program: str) -> str:
    courses, matched = cidx.by_program(program)
    if not courses:
        return f"课程库中未收录「{program}」涉及的课程。可先用 list_minors 确认辅修专业，或用 search_courses 检索课程。"
    lines = [f"「{matched}」涉及的课程（{len(courses)} 门）："]
    for c in courses:
        lines.append(cidx.summary(c))
    lines.append("需要某门课详情请用 get_course_detail。")
    return "\n".join(lines)


def build_tools(kb: KnowledgeBase, courses: CourseIndex | None = None) -> dict[str, Tool]:
    tools = [
        Tool(
            name="list_minors",
            description="列出知识库中全部辅修专业（按院系分组）。适合先了解有哪些专业可选。",
            func=lambda: _list_minors(kb),
        ),
        Tool(
            name="get_minor_detail",
            description=(
                "获取指定辅修专业的完整培养方案：学分要求、必修/限选/选修课程、先修课程、"
                "接纳人数、主修限制、申请条件等。"
            ),
            func=lambda name: _get_minor_detail(kb, name),
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "专业名称，如「计算机科学与技术」"}
                },
                "required": ["name"],
            },
        ),
        Tool(
            name="search_minors",
            description=(
                "按关键词在全部培养方案中检索（如「先修 微积分」「接纳人数」「人工智能」），"
                "返回相关专业与摘要。适合不确定选哪个专业时的初步筛选。"
            ),
            func=lambda query: _search_minors(kb, query),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索关键词，可用空格分隔多个词"}
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="get_regulations",
            description=(
                "获取《清华大学本科生辅修学士学位专业教学管理办法》：申请条件、修读要求、"
                "学位授予等校级管理规定。"
            ),
            func=lambda: _get_regulations(kb),
        ),
    ]

    # 课程查询工具（加载了课程库才注册）
    if courses is not None and len(courses):
        tools += [
            Tool(
                name="search_courses",
                description=(
                    "按关键词在课程库中检索课程（名称/英文名/简介/教学要求等），返回匹配课程的"
                    "课程号、院系、学分、先修要求与简介。适合了解某门课程的基本信息。"
                ),
                func=lambda query: _search_courses(courses, query),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "检索关键词，可用空格分隔多个词"}
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="get_course_detail",
                description=(
                    "获取某门课程的完整信息：课程号、名称、学分、学时、先修要求、课程简介、"
                    "教学目标、考核方式、教材、教师，以及所属的辅修培养方案。"
                ),
                func=lambda course: _get_course_detail(courses, course),
                parameters={
                    "type": "object",
                    "properties": {
                        "course": {"type": "string", "description": "课程名称或课程号，如「数据结构」或 30240184"}
                    },
                    "required": ["course"],
                },
            ),
            Tool(
                name="list_courses_for_minor",
                description=(
                    "列出某个辅修培养方案中涉及的课程（课程号、学分、先修要求、简介），"
                    "适合了解「这个辅修要修哪些课」。"
                ),
                func=lambda program: _list_courses_for_minor(courses, program),
                parameters={
                    "type": "object",
                    "properties": {
                        "program": {"type": "string", "description": "辅修培养方案名，如「计算机科学与技术专业辅修培养方案」"}
                    },
                    "required": ["program"],
                },
            ),
        ]
    return {t.name: t for t in tools}
