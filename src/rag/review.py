"""
Deterministic code review helpers for inline-style findings.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except Exception:  # pragma: no cover
    tomllib = None  # type: ignore[assignment]

try:
    import tomli
except Exception:  # pragma: no cover
    tomli = None  # type: ignore[assignment]

try:
    from pip._vendor import tomli as pip_tomli  # type: ignore
except Exception:  # pragma: no cover
    pip_tomli = None  # type: ignore[assignment]


@dataclass(frozen=True)
class ReviewFinding:
    path: str
    line: int
    severity: str
    message: str
    link: str = ""


@dataclass(frozen=True)
class ReviewTarget:
    mode: str
    path: Path | None = None
    start_line: int | None = None
    end_line: int | None = None


@dataclass(frozen=True)
class ReviewReport:
    mode: str
    label: str
    findings: tuple[ReviewFinding, ...]
    source_lines: tuple[tuple[int, str], ...] = ()
    summary: str = ""


@dataclass(frozen=True)
class OpenTarget:
    path: Path
    line: int | None = None


@dataclass(frozen=True)
class ReviewThread:
    thread_id: str
    path: str
    line: int
    status: str
    comments: tuple[str, ...]


_TEXT_FILE_EXTENSIONS = {
    "",
    ".asc",
    ".c",
    ".cc",
    ".cfg",
    ".conf",
    ".cpp",
    ".css",
    ".csv",
    ".go",
    ".h",
    ".hpp",
    ".html",
    ".java",
    ".js",
    ".json",
    ".md",
    ".py",
    ".rb",
    ".rs",
    ".sh",
    ".sql",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}


def _threads_file(project_root: Path) -> Path:
    return project_root / ".cache" / "review_threads.json"


def terminal_file_link(project_root: Path, path: str, line: int) -> str:
    abs_path = (project_root / path).resolve()
    target = f"file://{abs_path}#L{line}"
    label = f"{path}:{line}"
    return f"\x1b]8;;{target}\x1b\\{label}\x1b]8;;\x1b\\"


def load_threads(project_root: Path) -> list[ReviewThread]:
    threads_path = _threads_file(project_root)
    if not threads_path.exists():
        return []
    try:
        payload = json.loads(threads_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    threads: list[ReviewThread] = []
    for item in payload:
        threads.append(
            ReviewThread(
                thread_id=str(item.get("thread_id", "")),
                path=str(item.get("path", "")),
                line=int(item.get("line", 1)),
                status=str(item.get("status", "open")),
                comments=tuple(str(comment) for comment in item.get("comments", [])),
            )
        )
    return threads


def save_threads(project_root: Path, threads: list[ReviewThread]) -> None:
    threads_path = _threads_file(project_root)
    threads_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "thread_id": thread.thread_id,
            "path": thread.path,
            "line": thread.line,
            "status": thread.status,
            "comments": list(thread.comments),
        }
        for thread in threads
    ]
    threads_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def handle_thread_command(command: str, project_root: Path) -> str:
    raw = (command or "").strip()
    if raw.lower() == "threads":
        threads = load_threads(project_root)
        if not threads:
            return "No saved review threads."
        lines = ["Saved review threads"]
        for thread in threads:
            lines.append(
                f"- {thread.thread_id} {thread.path}:{thread.line} [{thread.status}] {thread.comments[-1] if thread.comments else ''}".rstrip()
            )
        return "\n".join(lines)

    add_match = re.fullmatch(
        r"thread\s+add\s+(?P<path>[^:]+):(?P<line>\d+)\s+(?P<comment>.+)",
        raw,
    )
    if add_match:
        rel_path = add_match.group("path")
        line = int(add_match.group("line"))
        comment = add_match.group("comment").strip()
        try:
            _resolve_project_path(project_root, Path(rel_path))
        except ValueError:
            return f"Thread path must stay inside the repo: {rel_path}"
        threads = load_threads(project_root)
        thread = ReviewThread(
            thread_id=f"t{int(time.time() * 1000)}",
            path=rel_path,
            line=line,
            status="open",
            comments=(comment,),
        )
        threads.append(thread)
        save_threads(project_root, threads)
        return f"Saved thread {thread.thread_id} on {rel_path}:{line}."

    reply_match = re.fullmatch(r"thread\s+reply\s+(?P<id>\S+)\s+(?P<comment>.+)", raw)
    if reply_match:
        threads = load_threads(project_root)
        for index, thread in enumerate(threads):
            if thread.thread_id == reply_match.group("id"):
                threads[index] = ReviewThread(
                    thread_id=thread.thread_id,
                    path=thread.path,
                    line=thread.line,
                    status=thread.status,
                    comments=thread.comments + (reply_match.group("comment").strip(),),
                )
                save_threads(project_root, threads)
                return f"Updated thread {thread.thread_id}."
        return f"Thread not found: {reply_match.group('id')}"

    resolve_match = re.fullmatch(r"thread\s+resolve\s+(?P<id>\S+)", raw)
    if resolve_match:
        threads = load_threads(project_root)
        for index, thread in enumerate(threads):
            if thread.thread_id == resolve_match.group("id"):
                threads[index] = ReviewThread(
                    thread_id=thread.thread_id,
                    path=thread.path,
                    line=thread.line,
                    status="resolved",
                    comments=thread.comments,
                )
                save_threads(project_root, threads)
                return f"Resolved thread {thread.thread_id}."
        return f"Thread not found: {resolve_match.group('id')}"

    return (
        "Use `threads`, `thread add <path:line> <comment>`, "
        "`thread reply <id> <comment>`, or `thread resolve <id>`."
    )


def parse_review_target(command: str) -> ReviewTarget:
    raw = (command or "").strip()
    lowered = raw.lower()
    if lowered == "review diff":
        return ReviewTarget(mode="diff")
    if lowered == "review staged":
        return ReviewTarget(mode="staged")

    match = re.fullmatch(
        r"review\s+(?P<path>[^:]+?)(?::(?P<start>\d+)(?:-(?P<end>\d+))?)?\s*",
        raw,
    )
    if not match:
        raise ValueError(
            "Use `review diff`, `review staged`, or `review <path[:line[-line]]>`."
        )

    start_line = int(match.group("start")) if match.group("start") else None
    end_line = int(match.group("end")) if match.group("end") else start_line
    if start_line and end_line and end_line < start_line:
        raise ValueError("Review range end must be greater than or equal to start.")

    return ReviewTarget(
        mode="file",
        path=Path(match.group("path")),
        start_line=start_line,
        end_line=end_line,
    )


def review_command(command: str, project_root: Path) -> str:
    try:
        target = parse_review_target(command)
    except ValueError as exc:
        return str(exc)
    if target.mode == "diff":
        return _review_diff(project_root)
    if target.mode == "staged":
        return _review_diff(project_root, staged=True)
    assert target.path is not None
    try:
        return _review_file_target(
            project_root, target.path, target.start_line, target.end_line
        )
    except ValueError:
        return f"Review path must stay inside the repo: {target.path}"


def parse_open_target(command: str) -> OpenTarget:
    raw = (command or "").strip()
    match = re.fullmatch(r"open\s+(?P<path>[^:]+?)(?::(?P<line>\d+))?\s*", raw)
    if not match:
        raise ValueError("Use `open <path[:line]>`.")
    return OpenTarget(
        path=Path(match.group("path")),
        line=int(match.group("line")) if match.group("line") else None,
    )


def build_review_report(command: str, project_root: Path) -> ReviewReport | None:
    try:
        target = parse_review_target(command)
    except ValueError:
        return None

    if target.mode != "file" or target.path is None:
        return None

    try:
        resolved = _resolve_project_path(project_root, target.path)
    except ValueError:
        return None
    if (
        not resolved.exists()
        or resolved.is_dir()
        or not _is_supported_text_file(resolved)
    ):
        return None

    findings = _analyze_file(resolved, project_root)
    rel_path = resolved.relative_to(project_root).as_posix()

    start_line = target.start_line
    end_line = target.end_line
    if start_line is not None and end_line is not None:
        findings = [
            finding for finding in findings if start_line <= finding.line <= end_line
        ]
        excerpt_start = start_line
        excerpt_end = end_line
        label = f"{rel_path}:{start_line}-{end_line}"
    else:
        if findings:
            excerpt_start = max(1, min(finding.line for finding in findings) - 2)
            excerpt_end = max(finding.line for finding in findings) + 2
        else:
            excerpt_start = 1
            excerpt_end = 20
        label = rel_path

    source_lines = _read_source_excerpt(resolved, excerpt_start, excerpt_end)
    summary = (
        f"{len(findings)} finding(s)"
        if findings
        else f"No review findings for {label}."
    )
    return ReviewReport(
        mode="file",
        label=label,
        findings=tuple(
            sorted(
                findings, key=lambda item: (item.line, _severity_rank(item.severity))
            )
        ),
        source_lines=tuple(source_lines),
        summary=summary,
    )


def build_open_report(command: str, project_root: Path) -> ReviewReport | None:
    try:
        target = parse_open_target(command)
        resolved = _resolve_project_path(project_root, target.path)
    except ValueError:
        return None
    if (
        not resolved.exists()
        or resolved.is_dir()
        or not _is_supported_text_file(resolved)
    ):
        return None

    rel_path = resolved.relative_to(project_root).as_posix()
    if target.line is None:
        start_line = 1
        end_line = 20
        label = rel_path
    else:
        start_line = max(1, target.line - 3)
        end_line = target.line + 3
        label = f"{rel_path}:{target.line}"

    source_lines = _read_source_excerpt(resolved, start_line, end_line)
    return ReviewReport(
        mode="open",
        label=label,
        findings=(),
        source_lines=tuple(source_lines),
        summary="Source excerpt",
    )


def _review_diff(project_root: Path, staged: bool = False) -> str:
    changed_lines_by_file = _changed_lines_from_git_diff(project_root, staged=staged)
    if not changed_lines_by_file:
        diff_label = "`git diff --staged`" if staged else "`git diff HEAD`"
        return f"No changed lines to review in {diff_label}."

    findings: list[ReviewFinding] = []
    unsupported: list[str] = []

    for rel_path, changed_lines in sorted(changed_lines_by_file.items()):
        abs_path = project_root / rel_path
        if not abs_path.exists():
            continue
        if not _is_supported_text_file(abs_path):
            unsupported.append(rel_path)
            continue
        file_findings = _analyze_file(abs_path, project_root)
        findings.extend(
            finding for finding in file_findings if finding.line in changed_lines
        )

    if findings:
        return _format_findings(findings, heading="Review findings for changed lines")

    if unsupported:
        sample = ", ".join(unsupported[:3])
        return (
            "No review findings on changed Python lines. "
            f"Skipped unsupported files: {sample}."
        )

    return "No review findings on changed lines."


def _review_file_target(
    project_root: Path,
    raw_path: Path,
    start_line: int | None,
    end_line: int | None,
) -> str:
    resolved = _resolve_project_path(project_root, raw_path)
    if not resolved.exists():
        return f"File not found: {raw_path}"
    if resolved.is_dir():
        return f"Expected a file, got a directory: {raw_path}"
    if not _is_supported_text_file(resolved):
        return f"Review currently supports common text/code files only: {raw_path}"

    findings = _analyze_file(resolved, project_root)
    rel_path = resolved.relative_to(project_root).as_posix()

    if start_line is not None and end_line is not None:
        findings = [
            finding for finding in findings if start_line <= finding.line <= end_line
        ]
        range_label = f"{rel_path}:{start_line}-{end_line}"
    else:
        range_label = rel_path

    if not findings:
        return f"No review findings for {range_label}."

    return _format_findings(findings, heading=f"Review findings for {range_label}")


def _resolve_project_path(project_root: Path, raw_path: Path) -> Path:
    candidate = raw_path if raw_path.is_absolute() else project_root / raw_path
    resolved = candidate.resolve()
    resolved.relative_to(project_root.resolve())
    return resolved


def _changed_lines_from_git_diff(
    project_root: Path, staged: bool = False
) -> dict[str, set[int]]:
    try:
        command = ["git", "diff", "--unified=0", "--no-color"]
        if staged:
            command.append("--staged")
        else:
            command.append("HEAD")
        command.append("--")
        result = subprocess.run(
            command,
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return {}

    if result.returncode not in {0, 1} or not result.stdout.strip():
        return {}

    changed: dict[str, set[int]] = {}
    current_file: str | None = None
    for line in result.stdout.splitlines():
        if line.startswith("+++ b/"):
            current_file = line[6:]
            changed.setdefault(current_file, set())
            continue
        if not current_file or not line.startswith("@@"):
            continue
        match = re.search(r"\+(\d+)(?:,(\d+))?", line)
        if not match:
            continue
        start = int(match.group(1))
        count = int(match.group(2) or "1")
        if count <= 0:
            continue
        changed[current_file].update(range(start, start + count))
    return changed


def _analyze_python_file(path: Path, project_root: Path) -> list[ReviewFinding]:
    try:
        source = path.read_text(encoding="utf-8")
    except Exception as exc:
        rel_path = path.relative_to(project_root).as_posix()
        return [
            ReviewFinding(
                path=rel_path,
                line=1,
                severity="medium",
                message=f"Could not read file for review: {exc}",
            )
        ]

    rel_path = path.relative_to(project_root).as_posix()
    try:
        tree = ast.parse(source, filename=rel_path)
    except SyntaxError as exc:
        return [
            ReviewFinding(
                path=rel_path,
                line=max(1, int(exc.lineno or 1)),
                severity="high",
                message=f"Syntax error prevents reliable execution: {exc.msg}.",
            )
        ]

    visitor = _PythonReviewVisitor(rel_path, project_root)
    visitor.visit(tree)
    return visitor.findings


def _analyze_file(path: Path, project_root: Path) -> list[ReviewFinding]:
    suffix = path.suffix.lower()
    if suffix == ".py":
        return _analyze_python_file(path, project_root)
    if suffix == ".json":
        return _analyze_json_file(path, project_root)
    if suffix == ".toml":
        return _analyze_toml_file(path, project_root)
    return _analyze_text_file(path, project_root)


def _analyze_json_file(path: Path, project_root: Path) -> list[ReviewFinding]:
    rel_path = path.relative_to(project_root).as_posix()
    source = path.read_text(encoding="utf-8")
    try:
        json.loads(source)
    except json.JSONDecodeError as exc:
        return [
            ReviewFinding(
                path=rel_path,
                line=max(1, int(exc.lineno)),
                severity="high",
                message=f"Invalid JSON: {exc.msg}.",
                link=terminal_file_link(
                    project_root, rel_path, max(1, int(exc.lineno))
                ),
            )
        ]
    return _scan_text_patterns(source.splitlines(), rel_path, project_root)


def _analyze_toml_file(path: Path, project_root: Path) -> list[ReviewFinding]:
    rel_path = path.relative_to(project_root).as_posix()
    source = path.read_text(encoding="utf-8")
    toml_loader = _toml_loads()
    if toml_loader is not None:
        try:
            toml_loader(source)
        except Exception as exc:
            line = _extract_error_line(str(exc))
            return [
                ReviewFinding(
                    path=rel_path,
                    line=line,
                    severity="high",
                    message=f"Invalid TOML: {exc}.",
                    link=terminal_file_link(project_root, rel_path, line),
                )
            ]
    return _scan_text_patterns(source.splitlines(), rel_path, project_root)


def _analyze_text_file(path: Path, project_root: Path) -> list[ReviewFinding]:
    rel_path = path.relative_to(project_root).as_posix()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        return [
            ReviewFinding(
                path=rel_path,
                line=1,
                severity="medium",
                message=f"Could not read file for review: {exc}",
                link=terminal_file_link(project_root, rel_path, 1),
            )
        ]
    return _scan_text_patterns(lines, rel_path, project_root)


def _scan_text_patterns(
    lines: list[str], rel_path: str, project_root: Path
) -> list[ReviewFinding]:
    findings: list[ReviewFinding] = []
    for lineno, line in enumerate(lines, start=1):
        stripped = line.strip()
        if "todo security" in stripped.lower():
            findings.append(
                _finding(
                    rel_path,
                    lineno,
                    "medium",
                    "Security-sensitive TODO left in source; behavior may be incomplete.",
                    project_root,
                )
            )
        if "password=" in stripped.lower() or "api_key" in stripped.lower():
            findings.append(
                _finding(
                    rel_path,
                    lineno,
                    "high",
                    "Possible hardcoded credential in source or config.",
                    project_root,
                )
            )
        if re.search(r"\beval\s*\(", stripped):
            findings.append(
                _finding(
                    rel_path,
                    lineno,
                    "high",
                    "`eval(...)` appears in source and can execute attacker-controlled code.",
                    project_root,
                )
            )
        if re.search(r"\binnerHTML\s*=", stripped):
            findings.append(
                _finding(
                    rel_path,
                    lineno,
                    "high",
                    "`innerHTML` assignment can create XSS risk with untrusted content.",
                    project_root,
                )
            )
        if re.search(r"\byaml\.load\s*\(", stripped):
            findings.append(
                _finding(
                    rel_path,
                    lineno,
                    "high",
                    "`yaml.load(...)` is unsafe with untrusted input unless a safe loader is enforced.",
                    project_root,
                )
            )
    return findings


def _format_findings(findings: list[ReviewFinding], heading: str) -> str:
    ordered = sorted(
        findings,
        key=lambda item: (
            _severity_rank(item.severity),
            item.path,
            item.line,
            item.message,
        ),
    )
    lines = [heading]
    for finding in ordered:
        lines.append(
            f"- {finding.path}:{finding.line} [{finding.severity}] {finding.message}"
        )
    return "\n".join(lines)


def _read_source_excerpt(
    path: Path, start_line: int, end_line: int
) -> list[tuple[int, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return [(1, "")]
    start = max(1, start_line)
    end = max(start, min(end_line, len(lines)))
    return [(lineno, lines[lineno - 1]) for lineno in range(start, end + 1)]


def _is_supported_text_file(path: Path) -> bool:
    return path.suffix.lower() in _TEXT_FILE_EXTENSIONS


def _severity_rank(severity: str) -> int:
    order = {"high": 0, "medium": 1, "low": 2}
    return order.get(severity, 99)


class _PythonReviewVisitor(ast.NodeVisitor):
    def __init__(self, rel_path: str, project_root: Path) -> None:
        self.rel_path = rel_path
        self.project_root = project_root
        self.findings: list[ReviewFinding] = []

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is None:
            self.findings.append(
                _finding(
                    self.rel_path,
                    node.lineno,
                    "medium",
                    "Bare `except:` catches `KeyboardInterrupt` and `SystemExit`, "
                    "which can hide real failures and block clean shutdown.",
                    self.project_root,
                )
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if self._is_subprocess_call(node) and self._has_shell_true(node):
            self.findings.append(
                _finding(
                    self.rel_path,
                    node.lineno,
                    "high",
                    "`subprocess` call uses `shell=True`, which executes through a "
                    "shell parser and is easy to abuse with untrusted input.",
                    self.project_root,
                )
            )

        if self._is_name(node.func, {"eval", "exec"}):
            self.findings.append(
                _finding(
                    self.rel_path,
                    node.lineno,
                    "high",
                    f"`{self._func_name(node.func)}` executes dynamic code and is a "
                    "real code-execution risk if any part of the input is attacker-controlled.",
                    self.project_root,
                )
            )

        if self._is_pickle_load(node):
            self.findings.append(
                _finding(
                    self.rel_path,
                    node.lineno,
                    "high",
                    "`pickle.load`/`pickle.loads` can execute attacker-controlled "
                    "code during deserialization.",
                    self.project_root,
                )
            )

        self.generic_visit(node)

    @staticmethod
    def _is_name(node: ast.AST, candidates: set[str]) -> bool:
        return isinstance(node, ast.Name) and node.id in candidates

    @staticmethod
    def _func_name(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return "call"

    @staticmethod
    def _is_subprocess_call(node: ast.Call) -> bool:
        if isinstance(node.func, ast.Attribute) and isinstance(
            node.func.value, ast.Name
        ):
            return node.func.value.id == "subprocess" and node.func.attr in {
                "run",
                "Popen",
                "call",
                "check_call",
                "check_output",
            }
        return False

    @staticmethod
    def _has_shell_true(node: ast.Call) -> bool:
        for keyword in node.keywords:
            if keyword.arg == "shell" and isinstance(keyword.value, ast.Constant):
                return keyword.value.value is True
        return False

    @staticmethod
    def _is_pickle_load(node: ast.Call) -> bool:
        return (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "pickle"
            and node.func.attr in {"load", "loads"}
        )


def _finding(
    rel_path: str,
    line: int,
    severity: str,
    message: str,
    project_root: Path,
) -> ReviewFinding:
    return ReviewFinding(
        path=rel_path,
        line=line,
        severity=severity,
        message=message,
        link=terminal_file_link(project_root, rel_path, line),
    )


def _extract_error_line(message: str) -> int:
    match = re.search(r"line (\d+)", message)
    return int(match.group(1)) if match else 1


def _toml_loads() -> "typing.Callable[[str], typing.Any] | None":
    if tomllib is not None:
        return tomllib.loads
    if tomli is not None:
        return tomli.loads
    if pip_tomli is not None:
        return pip_tomli.loads
    return None
