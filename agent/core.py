import json, re
import httpx
from .memory import ShortTermMemory, LongTermMemory
from .tools import Tools
from .planner import CoursePlanner
from .prompts import SYSTEM_PROMPT
from .data_loader import load_minors


class MinorAdvisorAgent:
    """The main agent orchestrator for Tsinghua Minor Program advising."""

    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com/v1"):
        self.api_key = api_key
        self.base_url = base_url

        # Load long-term memory (minor program database)
        minors = load_minors()
        self.ltm = LongTermMemory(minors)

        # Initialize tools
        self.tools = Tools(self.ltm, api_key=api_key)

        # Initialize planner
        self.planner = CoursePlanner(api_key, base_url)

        # Initialize short-term memory (per-session, will be copied for each session)
        self._default_stm = ShortTermMemory()

    def create_session(self) -> "AgentSession":
        """Create a new conversation session."""
        return AgentSession(
            api_key=self.api_key,
            base_url=self.base_url,
            ltm=self.ltm,
            tools=self.tools,
            planner=self.planner
        )


class AgentSession:
    """A single conversation session with its own short-term memory."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        ltm: LongTermMemory,
        tools: Tools,
        planner: CoursePlanner
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.ltm = ltm
        self.tools = tools
        self.planner = planner
        self.stm = ShortTermMemory()
        self.stm.add("system", SYSTEM_PROMPT)

    def process_message(self, user_message: str) -> str:
        """Process a user message and return the agent response."""
        self.stm.add("user", user_message)

        # Get response from LLM
        response = self._call_llm()
        self.stm.add("assistant", response)
        return response

    def process_with_planning(self, major: str, grade: str, minor_name: str) -> str:
        """Generate a course plan for a specific minor."""
        plan = self.planner.generate_plan(major, grade, minor_name, self.ltm)
        self.stm.add("user", f"请为{major}专业{grade}学生制定{minor_name}辅修修读计划")
        self.stm.add("assistant", plan)
        return plan

    def get_history(self) -> list[dict]:
        return self.stm.to_llm_format()

    def clear(self):
        self.stm.clear()
        self.stm.add("system", SYSTEM_PROMPT)

    def _call_llm(self, temperature: float = 0.3) -> str:
        """Call DeepSeek API and handle tool use via prompt-based function calling."""
        messages = self.stm.to_llm_format()

        # Build the tool-augmented system prompt
        tool_descriptions = self.tools.get_tool_descriptions()
        tool_block = "\n\n你可以在回答前使用以下工具获取信息。如果需要使用工具，输出格式为：\n"
        tool_block += "THOUGHT: <你的思考过程>\n"
        tool_block += "ACTION: <工具名称>\n"
        tool_block += "PARAMS: {\"参数名\": \"参数值\"}\n\n"
        tool_block += "工具列表：\n"
        for t in tool_descriptions:
            tool_block += f"- {t['name']}: {t['description']}\n"
            if t['parameters']:
                for pname, pinfo in t['parameters'].items():
                    tool_block += f"  参数 {pname}: {pinfo.get('description', '')}\n"

        # Add tool instructions to the last system message
        augmented_messages = []
        for msg in messages:
            if msg["role"] == "system":
                augmented_messages.append({
                    "role": "system",
                    "content": msg["content"] + "\n" + tool_block
                })
            else:
                augmented_messages.append(msg)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "deepseek-chat",
            "messages": augmented_messages,
            "temperature": temperature,
            "max_tokens": 4096
        }

        with httpx.Client(timeout=90) as client:
            resp = client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]

        # Check if the LLM wants to use a tool
        tool_result = self._parse_tool_call(content)
        if tool_result:
            tool_name, params = tool_result
            result = self._execute_tool(tool_name, params)
            self.stm.add("tool", result, tool_name=tool_name)
            return self._call_llm(temperature)

        # Strip internal reasoning prefix before returning to user
        content = self._clean_response(content)
        return content

    def _clean_response(self, content: str) -> str:
        """Remove internal THOUGHT: prefix if present and not followed by a tool call."""
        lines = content.split("\n")
        cleaned = []
        skip_thought = False
        for line in lines:
            if line.strip().startswith("THOUGHT:") and not any(
                l.strip().startswith("ACTION:") for l in lines
            ):
                skip_thought = True
                continue
            if skip_thought and line.strip().startswith("THOUGHT:"):
                continue
            skip_thought = False
            cleaned.append(line)
        result = "\n".join(cleaned).strip()
        return result if result else content

    def _parse_tool_call(self, content: str):
        """Parse tool call from LLM output."""
        action_match = re.search(r"ACTION:\s*(\w+)", content)
        params_match = re.search(r"PARAMS:\s*(\{.*?\})", content, re.DOTALL)
        if action_match:
            tool_name = action_match.group(1)
            params = {}
            if params_match:
                try:
                    params = json.loads(params_match.group(1))
                except json.JSONDecodeError:
                    pass
            return tool_name, params
        return None

    def _execute_tool(self, tool_name: str, params: dict) -> str:
        """Execute a tool and return its result."""
        tool_map = {
            "list_minors": lambda: self.tools.list_minors(),
            "search_minors": lambda: self.tools.search_minors(params.get("keyword", "")),
            "get_minor_detail": lambda: self.tools.get_minor_detail(params.get("name", "")),
            "check_eligibility": lambda: self.tools.check_eligibility(
                params.get("major", ""), params.get("minor_name", "")
            ),
            "semantic_search": lambda: self.tools.semantic_search(params.get("query", "")),
        }
        func = tool_map.get(tool_name)
        if func:
            return func()
        return f"未知工具: {tool_name}"
