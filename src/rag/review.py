"""
Deterministic code review helpers for inline-style findings.
"""

from __future__ import annotations

import ast
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReviewFinding:
    path: str
    line: int
    severity: str
    message: str


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
        raise ValueError("Use `review diff`, `review staged`, or `review <path[:line[-line]]>`.")

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
    if not resolved.exists() or resolved.is_dir() or resolved.suffix != ".py":
        return None

    findings = _analyze_python_file(resolved, project_root)
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
        f"{len(findings)} finding(s)" if findings else f"No review findings for {label}."
    )
    return ReviewReport(
        mode="file",
        label=label,
        findings=tuple(sorted(findings, key=lambda item: (item.line, _severity_rank(item.severity)))),
        source_lines=tuple(source_lines),
        summary=summary,
    )


def build_open_report(command: str, project_root: Path) -> ReviewReport | None:
    try:
        target = parse_open_target(command)
        resolved = _resolve_project_path(project_root, target.path)
    except ValueError:
        return None
    if not resolved.exists() or resolved.is_dir() or not _is_supported_text_file(resolved):
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
        if abs_path.suffix != ".py":
            unsupported.append(rel_path)
            continue
        file_findings = _analyze_python_file(abs_path, project_root)
        findings.extend(
            finding for finding in file_findings if finding.line in changed_lines
        )

    if findings:
        return _format_findings(
            findings, heading="Review findings for changed lines"
        )

    if unsupported:
        sample = ", ".join(unsupported[:3])
        return (
            "No review findings on changed Python lines. "
            f"Skipped non-Python files: {sample}."
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
    if resolved.suffix != ".py":
        return f"Review currently supports Python files only: {raw_path}"

    findings = _analyze_python_file(resolved, project_root)
    rel_path = resolved.relative_to(project_root).as_posix()

    if start_line is not None and end_line is not None:
        findings = [
            finding
            for finding in findings
            if start_line <= finding.line <= end_line
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

    visitor = _PythonReviewVisitor(rel_path)
    visitor.visit(tree)
    return visitor.findings


def _format_findings(findings: list[ReviewFinding], heading: str) -> str:
    ordered = sorted(
        findings,
        key=lambda item: (_severity_rank(item.severity), item.path, item.line, item.message),
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
    def __init__(self, rel_path: str) -> None:
        self.rel_path = rel_path
        self.findings: list[ReviewFinding] = []

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is None:
            self.findings.append(
                ReviewFinding(
                    path=self.rel_path,
                    line=node.lineno,
                    severity="medium",
                    message=(
                        "Bare `except:` catches `KeyboardInterrupt` and `SystemExit`, "
                        "which can hide real failures and block clean shutdown."
                    ),
                )
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if self._is_subprocess_call(node) and self._has_shell_true(node):
            self.findings.append(
                ReviewFinding(
                    path=self.rel_path,
                    line=node.lineno,
                    severity="high",
                    message=(
                        "`subprocess` call uses `shell=True`, which executes through a "
                        "shell parser and is easy to abuse with untrusted input."
                    ),
                )
            )

        if self._is_name(node.func, {"eval", "exec"}):
            self.findings.append(
                ReviewFinding(
                    path=self.rel_path,
                    line=node.lineno,
                    severity="high",
                    message=(
                        f"`{self._func_name(node.func)}` executes dynamic code and is a "
                        "real code-execution risk if any part of the input is attacker-controlled."
                    ),
                )
            )

        if self._is_pickle_load(node):
            self.findings.append(
                ReviewFinding(
                    path=self.rel_path,
                    line=node.lineno,
                    severity="high",
                    message=(
                        "`pickle.load`/`pickle.loads` can execute attacker-controlled "
                        "code during deserialization."
                    ),
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
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            return (
                node.func.value.id == "subprocess"
                and node.func.attr in {"run", "Popen", "call", "check_call", "check_output"}
            )
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
