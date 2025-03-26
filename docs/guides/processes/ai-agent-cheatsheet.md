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

## TDD Process Structure

All development must follow this Test-Driven Development process:

### 1. RED Phase
- Write failing tests first
- Tests should clearly define expected behavior
- Commit these tests before implementation

### 2. GREEN Phase
- Implement minimal code to make tests pass
- Focus on functionality, not optimization
- Ensure all tests pass consistently

### 3. REFACTOR Phase
- Clean up and optimize the implementation
- Maintain passing tests throughout refactoring
- Improve code quality, performance, and readability

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

## Documentation Structure

Each task should include these standard sections with emoji markers:

- 🎯 **Overview**: Background, objective, and scope
- 📐 **Technical Requirements**: Architecture, schema, API design
- 🧪 **Testing Requirements**: TDD process, test cases, verification
- 📄 **Documentation Requirements**: Files to update, content to add
- 🛠️ **Implementation Process**: Step-by-step workflow
- ✅ **Acceptance Criteria**: Requirements for completion

## Structured Test Cases

Format test cases with checkboxes and clear descriptions:

```markdown
### Test Cases

- [ ] Test `test_team_schema_validation`: Verify team schema validates correctly
- [ ] Test `test_create_team`: Verify team creation stores data correctly
- [ ] Test `test_retrieve_team`: Verify team retrieval returns correct data
```

## Implementation Examples

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
- Format tasks with standard emoji section markers
- Provide structured acceptance criteria with checkboxes

### ❌ Don't

- Skip writing tests
- Use pip/poetry/conda directly
- Use Pandas instead of Polars
- Create root-level directories
- Bypass linting or pre-commit hooks
- Implement hard-coded values (use configuration)
- Write overly detailed implementation examples
- Implement features without clear acceptance criteria

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

## Acceptance Criteria Pattern

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

## Feature Implementation Checklist

1. Create failing tests in the appropriate test directory
2. Implement feature in the correct module
3. Ensure all tests pass, including existing ones
4. Add appropriate documentation
5. Run pre-commit hooks
6. Verify against architecture guidelines

## Remember

This project strictly follows test-driven development practices. Never implement features without first writing tests that define the expected behavior. 