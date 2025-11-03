# CLI Policy and Standards

This document defines the **command-line interface (CLI)** policies and standards for the **RAG Transformer** project.

---

## Core Principles

### 1. User Experience

* **Intuitive**: Commands must follow common conventions.
* **Consistent**: Similar operations use similar syntax.
* **Helpful**: Clear errors, examples, and suggestions.
* **Accessible**: Support all experience levels and terminal environments.

### 2. POSIX Compliance

* Follow POSIX standards for argument parsing.
* Support short (`-h`) and long (`--help`) options.
* Use standard exit codes (`0` success, non-zero error).

### 3. Backwards Compatibility

* Maintain compatibility with existing commands.
* Deprecate features gracefully with migration paths.
* Version breaking changes properly.

---

## Command Structure

### Entry Points

1. **`rag`** – Main assistant (interactive)
2. **`rag-tui`** – Text User Interface mode
3. **`rag-collect`** – Data collection utility

### Standard Options

All commands must support:

```
--help, -h       Show help
--version        Show version
--verbose, -v    Enable verbose output
--quiet, -q      Suppress non-essential output
--no-color       Disable colored output
```

### Example (`rag`)

```bash
rag [OPTIONS] [--query QUERY]

Options:
  --query TEXT        Run a single query and exit
  --verbose, -v       Verbose mode
  --quiet, -q         Quiet mode
  --no-color          Disable color output
  --help, -h          Show help
  --version           Show version
```

---

## Output Standards

### Success

* Use positive, clear messages.
* Provide next steps where relevant.
* Emojis optional — disabled in `--no-color` mode.

### Errors

* Write to **stderr**.
* Format: `Error: <description>`
* Exit codes:

  * `0`: Success
  * `1`: General error
  * `2`: Misuse or invalid argument
* Suggest corrective actions when possible.

### Verbose Mode

* Show internal steps, timing, and debug info.
* Include stack traces for detailed error contexts.

---

## Interactive Mode

### Prompts

* Use clear prompt (`❯` or `>`) and feedback indicators.
* Support `Ctrl+C`, `Ctrl+D` for exit.

### Recognized Commands

```
exit | quit | q     Exit program
help | h            Show help
clear               Clear the screen
```

### Tool Syntax

```
CALC: <expression>
WIKI: <topic>
TIME:
```

---

## Environment Integration

### Environment Variables

* `NO_COLOR` – Disable color output
* `COLUMNS` – Terminal width
* `RAG_*` – Custom configuration variables

### Config Files

* Support `.env` for development.
* Follow XDG Base Directory standards.
* Provide clear examples in documentation.

---

## Accessibility

* `--no-color` disables colors and emojis.
* Respect the `NO_COLOR` standard ([no-color.org](https://no-color.org/)).
* Disable color when not writing to a TTY.
* Use semantic colors (green = success, red = error).
* Maintain high contrast and simple language.

---

## Testing Standards

* Test all command-line options and combinations.
* Verify correct exit codes and help text.
* Test both interactive and non-interactive behavior.
* Ensure functionality in CI/CD environments.

---

## Documentation Standards

* Keep help text concise, complete, and example-driven.
* Maintain consistent formatting across commands.
* Sync CLI help with man pages if provided.

---

## Security

* Sanitize user inputs and validate paths.
* Avoid logging sensitive data (API keys, secrets).
* Support secure environment variable injection.
* Document best practices for secret management.

---

## Deprecation and Migration

* Follow semantic versioning for CLI changes.
* Warn users about deprecations and offer migration guides.
* Support deprecated options for one major version before removal.

---

## Examples

### Error Example

```bash
$ rag --invalid-option
Error: Unknown option '--invalid-option'
Try 'rag --help' for usage information.
```

### Success Example

```bash
$ rag-collect
✅ Collected 1,234 documents
📊 Knowledge base updated with ML, sci-fi, and space data
```

### No-Color Example

```bash
$ rag --no-color --query "Hello"
Hello! How can I help you today?
```

---

## Compliance Checklist

Before releasing CLI changes:

* [ ] All commands support `--help` and `--version`
* [ ] Errors are clear and actionable
* [ ] Exit codes follow standards
* [ ] Interactive mode handles `Ctrl+C`
* [ ] Non-interactive behavior is validated
* [ ] Verbose mode provides debug details
* [ ] Help text includes examples
* [ ] All new options documented
* [ ] Backwards compatibility maintained
* [ ] Tests cover all CLI paths

---

**Review and Update:**
This policy must evolve with the project.
All contributors should review these standards before implementing CLI-related features.
