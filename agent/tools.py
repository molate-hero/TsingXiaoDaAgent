from typing import Optional
from .memory import LongTermMemory


class Tools:
    """Tools available to the agent."""

    def __init__(self, long_term_memory: LongTermMemory, api_key: str = ""):
        self.ltm = long_term_memory
        self._api_key = api_key

    def list_minors(self) -> str:
        """列出所有可用的辅修专业。"""
        names = self.ltm.list_all()
        return "清华大学2026年开放辅修专业列表：\n" + "\n".join(f"  {i+1}. {n}" for i, n in enumerate(names))

    def search_minors(self, keyword: str) -> str:
        """搜索辅修专业。"""
        results = self.ltm.search(keyword)
        if not results:
            return f"未找到与 '{keyword}' 相关的辅修专业。"
        lines = [f"找到 {len(results)} 个相关辅修专业："]
        for m in results:
            lines.append(f"\n【{m.name}】({m.department})")
            lines.append(f"  学分要求：{m.total_credits or '见培养方案'}")
            lines.append(f"  主修限制：{m.major_restrictions or '见培养方案'}")
            if m.capacity:
                lines.append(f"  接纳人数：{m.capacity}")
            if m.contact:
                lines.append(f"  咨询电话：{m.contact}")
        return "\n".join(lines)

    def get_minor_detail(self, name: str) -> str:
        """获取某个辅修专业的详细信息。"""
        prog = self.ltm.find_minor(name)
        if not prog:
            return f"未找到辅修专业: {name}"
        parts = [
            f"【{prog.name}】",
            f"开设院系：{prog.department}",
            f"学分要求：{prog.total_credits or '见培养方案'}",
        ]
        if prog.prerequisites:
            parts.append(f"先修课程：{prog.prerequisites}")
        if prog.major_restrictions:
            parts.append(f"主修专业限制：{prog.major_restrictions}")
        if prog.capacity:
            parts.append(f"每年可接纳：{prog.capacity}")
        if prog.contact:
            parts.append(f"咨询电话：{prog.contact}")
        parts.append(f"\n--- 培养方案详情 ---\n{prog.raw_text[:3000]}")
        return "\n".join(parts)

    def check_eligibility(self, major: str, minor_name: str) -> str:
        """检查学生主修专业是否符合辅修申请条件。"""
        prog = self.ltm.find_minor(minor_name)
        if not prog:
            return f"未找到辅修专业: {minor_name}"
        restrictions = prog.major_restrictions or "未明确列出，建议直接咨询院系确认。"
        return (
            f"【资格检查】主修：{major} → 辅修：{prog.name}\n"
            f"限制说明：{restrictions}\n"
            f"注意：辅修学位要求与主修专业归属不同专业类，请确认二者的专业类关系。"
        )

    def multi_agent_search(self, major: str, interests: str, grade: str = "") -> str:
        """【Multi-Agent 协同搜索】使用多个专业子 Agent 协同分析学生需求，推荐最适配的辅修专业。"""
        try:
            from .multi_agent import MultiAgentSystem
            mas = MultiAgentSystem(api_key=self._api_key)
            profile = {"major": major, "grade": grade, "interests": interests}
            return mas.search_recommendations(profile, self.ltm)
        except Exception as e:
            return f"Multi-Agent 搜索出错: {e}"

    def get_tool_descriptions(self) -> list[dict]:
        """Return tool descriptions in a format usable by LLM."""
        return [
            {
                "name": "list_minors",
                "description": "列出所有清华大学2026年开放辅修专业",
                "parameters": {}
            },
            {
                "name": "search_minors",
                "description": "根据关键词搜索辅修专业",
                "parameters": {
                    "keyword": {"type": "string", "description": "搜索关键词，如专业名称、院系"}
                }
            },
            {
                "name": "get_minor_detail",
                "description": "获取某个辅修专业的详细培养方案",
                "parameters": {
                    "name": {"type": "string", "description": "辅修专业名称"}
                }
            },
            {
                "name": "check_eligibility",
                "description": "检查学生主修专业是否符合辅修申请条件",
                "parameters": {
                    "major": {"type": "string", "description": "学生的主修专业"},
                    "minor_name": {"type": "string", "description": "辅修专业名称"}
                }
            },
            {
                "name": "multi_agent_search",
                "description": "【Multi-Agent】使用多个AI专家协同分析，推荐最适配的辅修专业",
                "parameters": {
                    "major": {"type": "string", "description": "学生的主修专业"},
                    "interests": {"type": "string", "description": "学生的兴趣方向"},
                    "grade": {"type": "string", "description": "年级"}
                }
            }
        ]
