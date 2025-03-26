---
title: NCAA Basketball Analytics - AI Agent Cheat Sheet
description: Quick reference guide for AI coding agents working on the NCAA Basketball Analytics project
---

# AI Agent Cheat Sheet

## Project Overview

NCAA Basketball Analytics is a data pipeline and predictive modeling system for college basketball analytics, with a focus on game predictions and tournament performance. The system collects data from ESPN APIs, processes it, generates features, trains models, and visualizes predictions.

## ⚠️ Critical Requirements

1. **Test-Driven Development**: *Always* follow TDD principles (write tests first, then implement code to pass tests)
2. **Use UV**: All dependency management must use UV (not pip, poetry, or other tools)
3. **Use Polars**: Always use Polars for data manipulation (never Pandas unless explicitly required)
4. **Respect Structure**: Do not create new root-level directories or modify project structure
5. **Follow Linting**: Never bypass pre-commit hooks or linting rules

## Project Structure

```
ncaa-basketball-analytics/
├── airflow/                      # Airflow DAGs and configurations
├── config/                       # Configuration files (YAML)
├── data/                         # Data storage (raw, processed, features)
├── docs/                         # Documentation
├── logs/                         # Application logs
├── notebooks/                    # Exploratory notebooks
├── src/                          # Core source code
├── tests/                        # Test suite (mirrors src/ structure)
├── tmp/                          # Temporary files (for multi-line commands)
└── pyproject.toml                # Python dependencies and metadata
```

## Key Technical Components

- **Language**: Python 3.12
- **Database**: DuckDB (column-oriented analytical database)
- **Data Manipulation**: Polars (high-performance DataFrame library)
- **Orchestration**: Apache Airflow
- **ML Frameworks**: PyTorch, MLflow
- **Visualization**: Plotly Dash

## Development Workflow

1. **Write failing tests** for the feature you're implementing
2. Implement code to **make tests pass**
3. **Refactor** while keeping tests passing
4. Run `pre-commit` hooks before committing
5. Submit PR with documentation updates

## Where to Find Documentation

- **Architecture**: `/docs/architecture/` - System design and components
- **Guides**: `/docs/guides/` - How-tos and development guidelines
- **Development Phases**: `/docs/architecture/development-phases.md` - Project roadmap
- **Task Guide**: `/docs/guides/processes/task-writing-guide.md` - Task structure

## Do's and Don'ts

### ✅ Do

- Write tests before implementing features
- Use UV for virtual environments and package management
- Follow naming conventions in existing code
- Document all non-trivial code
- Adhere to the project's modular structure
- Use type annotations

### ❌ Don't

- Skip writing tests
- Use pip/poetry/conda directly
- Use Pandas instead of Polars
- Create root-level directories
- Bypass linting or pre-commit hooks
- Implement hard-coded values (use configuration)

## Common Operations

### Setting up environment and dependencies

```bash
# Create virtual environment
uv venv

# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies 
uv sync
```

### Running tests

```bash
# Run all tests
python -m pytest

# Run specific tests with verbose output
python -m pytest tests/path/to/test_file.py -v
```

### Multi-line Command Workaround

⚠️ **Important**: AI agents cannot execute command-line commands containing newlines. For commands requiring multi-line input (like git commit messages), use this workaround:

```bash
# INCORRECT - Will fail:
git commit -m "Fix bug in feature calculation

- Updated algorithm to handle edge cases
- Added additional validation"

# CORRECT - Use temporary file:
# 1. Create a temporary markdown file
echo "Fix bug in feature calculation

- Updated algorithm to handle edge cases  
- Added additional validation" > tmp/commit-message.md

# 2. Reference the file in your command
git commit -F tmp/commit-message.md
```

This applies only to terminal commands. Multi-line edits in files work normally.

### Data Access Patterns

- Access database: Use DuckDB with Polars for querying
- Features: Access via feature registry in `src/features/registry.py`
- Configuration: Load from YAML files in `config/` directory using Pydantic

## Feature Implementation Checklist

1. Create failing tests in the appropriate test directory
2. Implement feature in the correct module
3. Ensure all tests pass, including existing ones
4. Add appropriate documentation
5. Run pre-commit hooks
6. Verify against architecture guidelines

## Remember

This project strictly follows test-driven development practices. Never implement features without first writing tests that define the expected behavior. 