import re
from collections import defaultdict, deque
from dataclasses import dataclass, field


@dataclass
class Course:
    id: str
    name: str
    credits: float = 0
    semester: str = ""         # 秋/春/春秋/夏
    raw_prereqs: str = ""
    is_required: bool = True
    course_type: str = ""      # 必修/限选/选修
    department: str = ""


def parse_courses_from_table(html_table: str) -> list[Course]:
    """Parse an HTML table fragment to extract course information."""
    courses = []
    rows = re.findall(r"<tr>(.*?)</tr>", html_table, re.DOTALL)
    current_type = ""

    for row in rows:
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.DOTALL)
        cells = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]

        # Detect section headers like <td colspan="5">必修课程 28 学分</td>
        if len(cells) == 1:
            text = cells[0]
            if "必修" in text:
                current_type = "必修"
            elif "限选" in text:
                current_type = "限选"
            elif "选修" in text:
                current_type = "选修"
            continue

        if len(cells) >= 4:
            course_id = cells[0]
            # Skip non-course rows (headers, merged cells)
            if not re.match(r"^\d", course_id) and course_id not in {"新开课", "新开"}:
                continue

            name = cells[1] if len(cells) > 1 else ""

            credits = 0
            try:
                credits = float(cells[2]) if len(cells) > 2 else 0
            except ValueError:
                pass

            semester = cells[3] if len(cells) > 3 else ""

            raw_prereqs = cells[4] if len(cells) > 4 else ""

            if course_id or name:
                courses.append(Course(
                    id=course_id,
                    name=name,
                    credits=credits,
                    semester=semester,
                    raw_prereqs=raw_prereqs,
                    is_required=(current_type == "必修"),
                    course_type=current_type
                ))

    return courses


def build_prerequisite_graph(courses: list[Course]) -> tuple[dict[str, set[str]], dict[str, Course]]:
    """Build a DAG of course dependencies.
    Returns (adjacency_list, course_map).
    adjacency_list: course_name -> set of prerequisite course_names
    """
    adj: dict[str, set[str]] = {}
    course_map: dict[str, Course] = {}

    for c in courses:
        if c.name:
            course_map[c.name] = c
            adj.setdefault(c.name, set())

    # Parse prerequisite text to link courses
    for c in courses:
        if not c.raw_prereqs or not c.name:
            continue
        prereq_text = c.raw_prereqs
        # Remove common noise
        prereq_text = re.sub(r"<[^>]+>", "", prereq_text)
        prereq_text = re.sub(r"[、，,、]", " ", prereq_text)

        # Extract known course names from the prerequisite text
        for other_name in course_map:
            if other_name != c.name and other_name in prereq_text:
                adj.setdefault(c.name, set()).add(other_name)

    return adj, course_map


def topological_sort(courses: list[Course]) -> list[list[Course]]:
    """Generate a semester-by-semester plan using topological sort.
    Returns list of semesters, each containing courses to take that semester.
    Constraints: prerequisites must come before dependents.
    """
    adj, course_map = build_prerequisite_graph(courses)

    # Calculate in-degrees (number of prerequisites not yet taken)
    in_degree: dict[str, int] = {name: 0 for name in adj}
    for name, prereqs in adj.items():
        for prereq in prereqs:
            if prereq in in_degree:
                in_degree[name] += 1

    # Queue of courses with no remaining prerequisites
    queue = deque()
    for name, degree in in_degree.items():
        if degree == 0 and name in course_map:
            queue.append(name)

    plan = []
    taken = set()
    remaining = set(course_map.keys())

    SEMESTER_CYCLE = ["秋", "春", "夏", "秋", "春", "夏", "秋", "春", "夏"]
    semester_idx = 0

    while remaining:
        if not queue:
            # Circular dependency or isolated courses - add remaining
            for name in list(remaining):
                if name not in taken and name not in queue:
                    queue.append(name)
            if not queue:
                break

        semester_courses = []
        next_queue = deque()

        while queue:
            name = queue.popleft()
            if name in taken or name not in course_map:
                continue

            course = course_map[name]

            # Check semester compatibility
            target_sem = SEMESTER_CYCLE[semester_idx % len(SEMESTER_CYCLE)]
            if course.semester:
                offered_in_fall = "秋" in course.semester
                offered_in_spring = "春" in course.semester
                offered_in_summer = "夏" in course.semester
                if ((target_sem == "秋" and not offered_in_fall) or
                        (target_sem == "春" and not offered_in_spring) or
                        (target_sem == "夏" and not offered_in_summer)):
                    # A course marked "春秋" is offered in both terms.
                    if name not in next_queue:
                        next_queue.append(name)
                    continue

            semester_courses.append(course)
            taken.add(name)
            remaining.discard(name)

            # Reduce in-degree of dependents
            for other_name, prereqs in adj.items():
                if name in prereqs:
                    in_degree[other_name] -= 1
                    if in_degree[other_name] == 0 and other_name not in taken:
                        queue.append(other_name)

        # Also push deferred courses to next queue
        while next_queue:
            name = next_queue.popleft()
            if name not in taken:
                queue.append(name)

        if semester_courses:
            plan.append(semester_courses)
        else:
            # No courses could be scheduled this semester, force add one
            if queue:
                name = queue.popleft()
                if name in course_map:
                    plan.append([course_map[name]])
                    taken.add(name)
                    remaining.discard(name)
            elif remaining:
                name = next(iter(remaining))
                if name in course_map:
                    plan.append([course_map[name]])
                    taken.add(name)
                    remaining.discard(name)

        semester_idx += 1

    return plan


def format_plan(plan: list[list[Course]], student_grade: str = "大二") -> str:
    """Format the plan as a human-readable table."""
    grade_map = {"大一": 1, "大二": 2, "大三": 3, "大四": 4}
    current_grade_num = grade_map.get(student_grade, 2)
    semester_labels = ["秋", "春", "夏", "秋", "春", "夏", "秋", "春", "夏"]
    year_labels = ["大二", "大二", "大三", "大三", "大四", "大四"]

    lines = ["📋 **拓扑排序算法生成的最优修读计划**\n"]
    lines.append("| 学期 | 课程名称 | 学分 | 类型 |")
    lines.append("|------|----------|------|------|")

    for i, semester_courses in enumerate(plan):
        if i >= len(semester_labels):
            break
        year_idx = current_grade_num - 1 + (i // 3)
        if year_idx >= 4:
            break
        year_label = f"{['大一','大二','大三','大四'][year_idx]}"
        sem_label = semester_labels[i]
        header = f"**{year_label} {sem_label}**"
        lines.append(f"| **{year_label} {sem_label}** | | | |")
        for course in semester_courses:
            lines.append(f"| | {course.name} | {course.credits} | {course.course_type} |")
        lines.append("| | | | |")

    lines.append("\n*注：此计划由课程先修关系 DAG 拓扑排序生成，考虑了学期开课约束。*")
    return "\n".join(lines)
