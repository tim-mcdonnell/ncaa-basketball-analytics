---
title: NCAA Basketball Analytics - AI Agent Cheat Sheet
description: Quick reference guide for AI coding agents working on the NCAA Basketball Analytics project
---

# AI Agent Cheat Sheet

## Project Overview

NCAA Basketball Analytics is a data pipeline and predictive modeling system for college basketball analytics, with a focus on game predictions and tournament performance. The system collects data from ESPN APIs, processes it, generates features, trains models, and visualizes predictions.

## ⚠️ Critical Guardrails

1. **Test-Driven Development**: *Always* follow TDD principles (write tests first, then implement code to pass tests)
2. **Never Skip Tests**: Never use `@pytest.mark.skip` or similar to bypass tests
3. **Never Bypass Linting**: Never use `# noqa`, `# type: ignore` or similar markers to bypass linting rules
4. **Never Edit Core Config**: Do not modify `pyproject.toml` under any circumstances
5. **Use UV**: All dependency management must use UV (not pip, poetry, or other tools)
6. **Use Polars**: Always use Polars for data manipulation (never Pandas unless explicitly required)
7. **Respect Structure**: Do not create new root-level directories or modify project structure
8. **Follow Linting**: Never bypass pre-commit hooks or linting rules

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
└── pyproject.toml                # Python dependencies and metadata (DO NOT EDIT)
```

## Development Workflow

### TDD Process

All development must follow this Test-Driven Development process:

1. **RED**: Write failing tests first that clearly define expected behavior
2. **GREEN**: Implement minimal code to make tests pass (focus on functionality, not optimization)
3. **REFACTOR**: Clean up and optimize while maintaining passing tests

Example:
```python
# 1. RED: Write the test first
def test_team_repository_get_by_id():
    """Test retrieving a team by ID."""
    # Arrange
    repo = TeamRepository(":memory:")
    team_id = "MICH"

    # Act
    result = repo.get_by_id(team_id)

    # Assert
    assert result is not None
    assert result.team_id == team_id

# 2. GREEN: Implement code to make the test pass
# 3. REFACTOR: Optimize while keeping tests passing
```

### Implementation Process

1. **Write failing tests** in the appropriate test directory
2. **Implement code** to make tests pass in the correct module
3. **Refactor** while keeping tests passing
4. **Run pre-commit hooks** before committing
5. **Update documentation** as needed
6. **Verify against architecture guidelines**

### Commit Frequency

Make frequent, meaningful commits throughout task implementation:

1. **Commit on Working Milestones**: Don't wait until the entire task is complete to commit
2. **Commit After Each Significant Change**: Each time you complete a meaningful component or feature
3. **Small, Focused Commits**: Each commit should represent a single logical change
4. **Descriptive Commit Messages**: Clearly describe what the commit accomplishes
5. **Examples of Good Commit Points**:
   - After writing the initial failing tests
   - After implementing a component that passes tests
   - After refactoring for performance or readability
   - After completing documentation updates

## Technical Specifications

### Key Technical Components

- **Language**: Python 3.12
- **Database**: DuckDB (column-oriented analytical database)
- **Data Manipulation**: Polars (high-performance DataFrame library)
- **Orchestration**: Apache Airflow
- **ML Frameworks**: PyTorch, MLflow
- **Visualization**: Plotly Dash

### Data Access Patterns

- **Database**: Use DuckDB with Polars for querying
- **Features**: Access via feature registry in `src/features/registry.py`
- **Configuration**: Load from YAML files in `config/` directory using Pydantic

## Documentation Standards

### Task Documentation Structure

Each task should include these standard sections with emoji markers:

- 🎯 **Overview**: Background, objective, and scope
- 📐 **Technical Requirements**: Architecture, schema, API design
- 🧪 **Testing Requirements**: TDD process, test cases, verification
- 📄 **Documentation Requirements**: Files to update, content to add
- 🛠️ **Implementation Process**: Step-by-step workflow
- ✅ **Acceptance Criteria**: Requirements for completion

### Structured Test Cases

Format test cases with checkboxes and clear descriptions:

```markdown
### Test Cases

- [ ] Test `test_team_schema_validation`: Verify team schema validates correctly
- [ ] Test `test_create_team`: Verify team creation stores data correctly
- [ ] Test `test_retrieve_team`: Verify team retrieval returns correct data
```

### Acceptance Criteria Pattern

Always structure acceptance criteria with checkboxes:

```markdown
## ✅ Acceptance Criteria

- [ ] All specified tests pass, including integration tests
- [ ] Database schema is correctly implemented
- [ ] Repository pattern enables all required data operations
- [ ] Error handling correctly manages failure cases
- [ ] Documentation is complete and accurate
- [ ] Code passes all linting and type checking
```

### Implementation Examples

When providing example code:

1. Keep examples minimal (<20 lines)
2. Focus on API/interfaces, not full implementations
3. Include type hints and docstrings
4. Use placeholder comments for implementation details

Good example:
```python
def get_team_by_id(team_id: str) -> Optional[Team]:
    """
    Retrieve team by ID from the database.

    Args:
        team_id: Unique team identifier

    Returns:
        Team object if found, None otherwise
    """
    # Implementation would go here
    pass
```

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

## Project Documentation

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
- Format tasks with standard emoji section markers
- Make frequent, small, focused commits with clear messages
- Commit after completing each meaningful component

### ❌ Don't

- Skip writing tests
- Use pip/poetry/conda directly
- Use Pandas instead of Polars
- Edit pyproject.toml
- Skip tests with @pytest.mark.skip
- Bypass linting with # noqa or similar
- Create root-level directories
- Implement hard-coded values (use configuration)
- Implement features without clear acceptance criteria
- Wait until the end of a task to make a single large commit

## Remember

This project strictly follows test-driven development practices. Never implement features without first writing tests that define the expected behavior.
