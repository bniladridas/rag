# CLI Policy and Standards

This document outlines the command-line interface (CLI) policies and standards for the RAG Transformer project.

## Core Principles

### 1. User Experience First
- **Intuitive**: Commands should be self-explanatory and follow common conventions
- **Consistent**: Similar operations should use similar syntax across all commands
- **Helpful**: Provide clear error messages and helpful suggestions
- **Accessible**: Support users with different experience levels

### 2. POSIX Compliance
- Follow POSIX standards for argument parsing where applicable
- Support both short (`-h`) and long (`--help`) option formats
- Use standard exit codes (0 for success, non-zero for errors)

### 3. Backwards Compatibility
- Maintain compatibility with existing command structures
- Deprecate features gracefully with clear migration paths
- Version all breaking changes appropriately

## Command Structure

### Entry Points
The project provides three main CLI entry points:

1. **`rag`** - Main interactive assistant
2. **`rag-tui`** - Text User Interface mode
3. **`rag-collect`** - Data collection utility

### Argument Conventions

#### Standard Options
All commands should support these standard options:
- `--help`, `-h`: Display help information
- `--version`: Display version information
- `--verbose`, `-v`: Enable verbose output
- `--quiet`, `-q`: Suppress non-essential output

#### Main Command (`rag`)
```bash
rag [OPTIONS] [--query QUERY]

Options:
  --query TEXT         Ask a single question and exit
  --verbose, -v        Enable verbose output
  --quiet, -q          Suppress non-essential output
  --no-color          Disable colored output
  --help, -h          Show help message
  --version           Show version information
```

## Output Standards

### Success Messages
- Use positive, encouraging language
- Include relevant emojis for better UX (when appropriate)
- Provide clear next steps when applicable

### Error Handling
- **Exit Codes**: Use standard exit codes (0=success, 1=general error, 2=misuse)
- **Error Messages**: Write to stderr, not stdout
- **Format**: `Error: <clear description of what went wrong>`
- **Suggestions**: When possible, suggest how to fix the issue

### Verbose Mode
When `--verbose` is enabled:
- Show additional debugging information
- Display processing steps
- Include timing information for operations
- Show full stack traces for errors

## Interactive Mode Standards

### Prompt Design
- Use clear, recognizable prompts (`❯` for input)
- Provide visual feedback for different states
- Support common shell-like shortcuts (Ctrl+C, Ctrl+D)

### Command Recognition
Interactive mode should recognize:
- `exit`, `quit`, `q` - Exit the program
- `help`, `h` - Show help information
- `clear` - Clear the screen (when supported)

### Tool Integration
Built-in tools should follow consistent patterns:
- `CALC: <expression>` - Calculator
- `WIKI: <topic>` - Wikipedia search
- `TIME:` - Current time/date

## Environment Integration

### Environment Variables
Support standard environment variables:
- `NO_COLOR` - Disable colored output
- `COLUMNS` - Terminal width for formatting
- `RAG_*` - Application-specific configuration

### Configuration Files
- Support `.env` files for local development
- Respect XDG Base Directory specification when possible
- Provide clear configuration examples

## Accessibility

### Color and Formatting
- Provide `--no-color` option to disable colors and emojis
- Respect `NO_COLOR` environment variable (https://no-color.org/)
- Automatically disable colors when output is not to a TTY
- Use semantic colors (red for errors, green for success)
- Ensure sufficient contrast for readability
- Support terminal themes
- Emojis should be removed in no-color mode for better accessibility

### Text Output
- Use clear, simple language
- Avoid technical jargon in user-facing messages
- Provide examples in help text
- Support different terminal widths

## Testing Standards

### CLI Testing Requirements
- Test all command-line options and combinations
- Verify exit codes for different scenarios
- Test interactive mode flows
- Validate help text accuracy and formatting

### Non-Interactive Testing
- Ensure commands work in CI/CD environments
- Test with different terminal configurations
- Verify behavior when stdin is not a TTY

## Documentation Requirements

### Help Text
- Keep help text concise but complete
- Include practical examples
- Use consistent formatting
- Update help text with any CLI changes

### Man Pages
- Consider providing man pages for complex commands
- Follow standard man page conventions
- Keep synchronized with CLI help text

## Security Considerations

### Input Validation
- Sanitize all user inputs
- Validate file paths and prevent directory traversal
- Limit resource usage (memory, CPU, network)

### API Keys and Secrets
- Never log or display API keys
- Support secure environment variable injection
- Provide clear guidance on secret management

## Migration and Deprecation

### Breaking Changes
- Follow semantic versioning for CLI changes
- Provide migration guides for major changes
- Support deprecated options for at least one major version

### Feature Deprecation
- Warn users about deprecated features
- Provide clear timelines for removal
- Suggest alternative approaches

## Examples

### Good CLI Practices
```bash
# Clear, helpful error message
$ rag --invalid-option
Error: Unknown option '--invalid-option'
Try 'rag --help' for more information.

# Helpful success message
$ rag-collect
✅ Successfully collected 1,234 documents
📊 Knowledge base updated with ML, sci-fi, and cosmos data
```

### Interactive Mode Flow
```bash
$ rag
🤖 Agentic RAG Transformer - ML, Sci-Fi, and Cosmos Assistant
Type 'exit' to quit, 'help' for instructions

❯ What is machine learning?
💡 Machine learning is a subset of artificial intelligence...

❯ CALC: 2^10
💡 1024

❯ exit
👋 Goodbye!
```

### No-Color Mode Examples
```bash
# With colors (default)
$ rag --query "Hello"
💡 Hello! How can I help you today?

# Without colors (accessible mode)
$ rag --no-color --query "Hello"
Hello! How can I help you today?

# Using NO_COLOR environment variable
$ NO_COLOR=1 rag --query "Hello"
Hello! How can I help you today?

# Interactive mode without colors
$ rag --no-color
Agentic RAG Transformer - ML, Sci-Fi, and Cosmos Assistant
Type 'exit' to quit, 'help' for instructions

> What is machine learning?
Machine learning is a subset of artificial intelligence...

> exit
Goodbye!
👋 Goodbye!
```

## Compliance Checklist

Before releasing CLI changes, ensure:
- [ ] All commands support `--help` and `--version`
- [ ] Error messages are clear and actionable
- [ ] Exit codes follow standards
- [ ] Interactive mode handles Ctrl+C gracefully
- [ ] Non-interactive environments are detected properly
- [ ] Verbose mode provides useful debugging information
- [ ] Help text includes practical examples
- [ ] All new options are documented
- [ ] Backwards compatibility is maintained
- [ ] Tests cover new CLI functionality

---

This policy should be reviewed and updated as the project evolves. All contributors should familiarize themselves with these standards before making CLI-related changes.
