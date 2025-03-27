---
title: How to Write Effective Tasks for AI Coding Agents
description: Guidelines for creating clear, comprehensive task descriptions for AI coding agents in the NCAA Basketball Analytics Project
---

# How to Write Effective Tasks for AI Coding Agents

## Quick Reference

For experienced users, here are the key elements to include in any AI task:

- ✅ Explicit requirements with exact file paths and naming conventions
- ✅ Complete test specifications with test case names and expected outcomes
- ✅ Scope boundaries that define what is and isn't included
- ✅ Decision history capturing conversation-derived choices
- ✅ Context references to related components and examples
- ✅ Machine-verifiable acceptance criteria

## Introduction

This guide outlines best practices for writing clear, comprehensive task descriptions for AI coding agents. Following these guidelines will help ensure that your AI assistant correctly understands, implements, and validates your coding requirements.

## Core Principles

### 1. Be Explicit, Not Implicit
- State requirements directly rather than assuming the AI will infer them
- Include exact file paths, naming conventions, and structural requirements
- Avoid vague terminology or ambiguous directions

### 2. Structure for Discoverability
- Use clear section headings that signal their purpose
- Apply consistent formatting and visual hierarchy
- Employ checklists and code blocks for clarity
- Use emoji icons to distinguish different sections visually (🎯, 📐, 🧪, 📄, 🛠️, ✅)

### 3. Emphasize Verification
- Include explicit testing requirements with specific test cases
- Require real-world validation, not just unit tests
- Provide concrete verification steps for checking implementation
- Define clear acceptance criteria

### 4. Prioritize Architecture Alignment
- Explicitly define architectural requirements and constraints
- Specify database structures, file organization, and design patterns
- Prevent misalignment by stating what NOT to do when necessary
- Reference existing architecture documentation

## AI-Specific Task Elements

### 📝 Decision Capture Framework

For tasks derived from conversations, capture key decisions explicitly:

```markdown
## 📝 Decision History

**Origin Discussion:** [Brief reference to originating conversation]

**Key Decisions:**
1. **Database Choice:** Selected DuckDB over SQLite because [specific reasons]
   - Considered SQLite but rejected due to [specific reason]

2. **Authentication Approach:** Implemented JWT-based auth because [specific reasons]
   - Alternative OAuth approach wasn't chosen because [specific reason]

**Open Questions:**
- How should we handle [specific edge case]? Default to [recommendation] if not specified.
```

### 🔎 Assumed Knowledge Section

Document background knowledge the AI is expected to have:

```markdown
## 🔎 Assumed Knowledge

**Project Patterns:**
- Repository pattern as implemented in `src/data/teams/repository.py`
- Error handling approach following `src/utils/errors.py`

**Technical Concepts:**
- Understanding of DuckDB's asynchronous API
- Familiarity with pytest fixtures and mocking

**Code Conventions:**
- Snake_case for variables and functions
- PascalCase for classes
- Type annotations required for all public functions
```

### 📚 Progressive Detail Structure

Organize information from most critical to supplementary:

```markdown
## 📚 Implementation Hierarchy

**Must Have (P0):**
- Core functionality X that enables [primary use case]
- Critical validation for [specific data/input]

**Should Have (P1):**
- Enhanced error handling for [specific scenarios]
- Performance optimization for [specific operation]

**Nice to Have (P2):**
- Additional utility functions for [specific case]
- Extra validation for edge case [description]
```

### 🔄 Contextual References

Link the task to related components in the codebase:

```markdown
## 🔄 Related Components

**Similar Implementations:**
- `src/data/games/repository.py`: Similar repository pattern implementation
- `src/services/espn/client.py`: Example of API client structure to follow

**Dependent Components:**
- `src/api/endpoints/teams.py`: Will use this implementation
- `src/ui/components/TeamList.tsx`: Will consume this data

**Required Libraries:**
- `duckdb-engine==0.9.2`: For database operations
- `pydantic==2.4.2`: For data validation
```

### 🤖 Machine-Parsable Formats

Use consistent, parsable patterns for key elements:

```markdown
## 🤖 Implementation Guidelines

**File Paths:**
- `src/data/players/repository.py`: Repository implementation
- `tests/data/players/test_repository.py`: Repository tests

**Function Signatures:**
```python
def get_player_by_id(player_id: str) -> Optional[Player]:
    """Retrieve player by ID."""
```

**Database Schema:**
```sql
CREATE TABLE players (
    player_id VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL
);
```
```

### ✓ Task Completion Validation

Include a structured validation checklist:

```markdown
## ✓ Implementation Validation

**Code Quality Checks:**
- [ ] Passes `mypy` type checking with zero errors
- [ ] Passes `pylint` with score ≥ 9.0/10
- [ ] Line coverage ≥ 90% for all new code

**Functional Verification:**
- [ ] All specified unit tests pass
- [ ] Integration test with [specific scenario] passes
- [ ] Manual verification with [specific input] produces [expected output]

**Documentation Verification:**
- [ ] Public API is documented with docstrings
- [ ] README.md updated with usage examples
- [ ] Architecture diagram updated (if applicable)
```

## Essential Task Components

### 🎯 Task Overview Section
- Title: Clear, concise description of the task
- Background: Context explaining why this task matters
- Objective: Specific outcome to achieve
- Scope: Explicit boundaries of what should/shouldn't be implemented

**Example:**
```markdown
## 🎯 Overview

**Background:** Reliable data storage is critical for analytics pipeline integrity and model training.

**Objective:** Implement DuckDB-based data storage component for NCAA basketball statistics.

**Scope:** Create schema, data access layer, and validation, but NOT visualization or analytics.
```

### 📐 Technical Requirements Section
- Architecture requirements with specific file paths and naming conventions
- Database schema with actual SQL statements
- API endpoints with exact routes and response formats
- Dependencies and external integrations
- Performance and security considerations

**Example:**
```markdown
## 📐 Technical Requirements

### Database Schema

```sql
CREATE TABLE teams (
    team_id VARCHAR PRIMARY KEY,
    team_name VARCHAR NOT NULL,
    conference VARCHAR,
    division VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Repository Structure

src/data/storage/
├── __init__.py
├── db.py            # Database connection management
├── repository.py    # Repository pattern implementation
├── schema.py        # Schema definitions
└── validation.py    # Data validation functions
```

### 🧪 Testing Framework Section
- Test-Driven Development (TDD) approach with RED-GREEN-REFACTOR steps
- Specific test cases with names and expected outcomes
- Unit testing requirements
- Integration testing requirements
- Real-world data testing instructions

**Example:**
```markdown
## 🧪 Testing Requirements

### Test-Driven Development Process

1. **RED Phase**:
   - Write failing tests for schema validation
   - Create tests for repository CRUD operations
   - Develop tests for data validation functions

2. **GREEN Phase**:
   - Implement schema definitions that pass validation tests
   - Build repository functions that satisfy CRUD tests
   - Create validation functions that pass test cases

3. **REFACTOR Phase**:
   - Optimize database access patterns
   - Enhance error handling
   - Improve data validation efficiency

### Test Cases

- [ ] Test `test_team_schema_validation`: Verify team schema validates correctly
- [ ] Test `test_create_team`: Verify team creation stores data correctly
- [ ] Test `test_retrieve_team`: Verify team retrieval returns correct data
- [ ] Test `test_update_team`: Verify team updates are stored correctly
- [ ] Test `test_delete_team`: Verify team deletion removes data correctly
```

### 📄 Documentation Requirements Section
- Files to create or update
- Required content for documentation
- Code commenting standards
- Examples or templates to follow

**Example:**
```markdown
## 📄 Documentation Requirements

- [ ] Create README.md in src/data/storage/ with module overview
- [ ] Document schema design decisions in docs/architecture/data-storage.md
- [ ] Add docstrings to all public functions with parameters and return types
- [ ] Include usage examples for repository pattern in docs/guides/data-access.md
```

### 🛠️ Implementation Process Section
- Step-by-step workflow
- Breakdown of implementation phases
- Dependencies between steps
- Verification checkpoints

**Example:**
```markdown
## 🛠️ Implementation Process

1. Implement database connection management
2. Create schema definitions with validation
3. Implement repository pattern base class
4. Add specialized repositories for each entity
5. Implement data validation functions
6. Add integration tests with test database
```

### ✅ Acceptance Criteria Section
- Explicit list of requirements that must be met
- Testing success parameters
- Performance benchmarks
- Documentation completeness
- Code quality standards

**Example:**
```markdown
## ✅ Acceptance Criteria

- [ ] All specified tests pass
- [ ] Database schema is created correctly
- [ ] Repository pattern enables CRUD operations for all entities
- [ ] Data validation prevents invalid data from being stored
- [ ] Documentation is complete and accurate
- [ ] Code passes linting with zero errors
- [ ] Implementation follows project architecture guidelines
```

## Implementation Examples

When providing implementation examples in tasks, keep them minimal and focused. Examples should illustrate the pattern or approach without implementing the full solution. They should be:

1. Short and concise (generally <20 lines)
2. Illustrative of important patterns
3. Well-documented with comments
4. Clear interfaces with type hints

**Good Example - Repository Pattern:**
```python
from typing import Dict, List, Optional
import duckdb
from src.data.models import Team

class TeamRepository:
    """Repository for team data operations."""

    def __init__(self, db_path: str):
        """Initialize with database path."""
        self.db_path = db_path

    def get_by_id(self, team_id: str) -> Optional[Team]:
        """Retrieve team by ID."""
        # Implementation would go here
        pass

    def create(self, team: Team) -> str:
        """Create a new team record."""
        # Implementation would go here
        pass
```

**Bad Example - Too Detailed:**
```python
# Don't provide overly detailed implementation
# with all edge cases and error handling
# This is too much for a task description

class TeamRepository:
    def __init__(self, db_path: str):
        self.conn = duckdb.connect(db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS teams (
                team_id VARCHAR PRIMARY KEY,
                team_name VARCHAR NOT NULL,
                conference VARCHAR,
                division VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

    def get_by_id(self, team_id: str) -> Optional[Team]:
        try:
            result = self.conn.execute(
                "SELECT * FROM teams WHERE team_id = ?",
                [team_id]
            ).fetchone()
            if not result:
                return None
            return Team(
                team_id=result[0],
                team_name=result[1],
                conference=result[2],
                division=result[3]
            )
        except Exception as e:
            logging.error(f"Error retrieving team: {e}")
            raise RepositoryError(f"Failed to retrieve team: {e}")

    # More methods with full implementation...
```

## Test Case Format

When writing test cases, provide clear names and descriptions that indicate what's being tested and the expected outcome. Use checkboxes to make them trackable.

**Example Test Case Format:**
```markdown
- [ ] Test `test_team_schema_validation`: Verify team schema validates correctly
```

**Example Test Function (for illustration):**
```python
def test_team_schema_validation():
    """Test that team schema validates correctly."""
    # Arrange
    valid_team = {
        "team_id": "MICH",
        "team_name": "Michigan Wolverines"
    }
    invalid_team = {"team_name": "Michigan Wolverines"}

    # Act
    valid_result = validate_team(valid_team)
    invalid_result = validate_team(invalid_team)

    # Assert
    assert valid_result.is_valid is True
    assert invalid_result.is_valid is False
    assert "team_id" in invalid_result.errors
```

## AI Task Template Structure

```markdown
# Task: [Descriptive Title]

## 🎯 Overview
**Background:** [Context and why this matters]
**Objective:** [Specific goal to accomplish]
**Scope:** [Clear boundaries of the task]

## 📝 Decision History
**Origin Discussion:** [Reference to source conversation]
**Key Decisions:**
- Decision 1: [Specific choice with rationale]
- Decision 2: [Specific choice with rationale]

## 🔎 Assumed Knowledge
**Project Patterns:** [List of patterns to follow]
**Technical Concepts:** [Required technical understanding]
**Code Conventions:** [Style and naming conventions]

## 📐 Technical Requirements
### Architecture
- [Specific structural requirements]
- [File paths and naming conventions]
- [Design patterns to follow]

### Database
```sql
-- Exact schema definition
CREATE TABLE example (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL
);
```

### API Design
- Endpoint: `[Exact endpoint path]`
- Method: `[HTTP method]`
- Request/Response format: [Provide exact format]

## 🧪 Testing Requirements
### Test-Driven Development Process
1. **RED Phase**: [Describe tests to write first]
2. **GREEN Phase**: [Describe implementation to make tests pass]
3. **REFACTOR Phase**: [Describe optimization and cleanup]

### Test Cases
- [ ] Test `test_name_1`: [Description of test case]
- [ ] Test `test_name_2`: [Description of test case]

### Real-World Testing
- Run: `[Exact command to execute]`
- Verify: [Specific outcomes to check]

## 📄 Documentation Requirements
- [ ] Update `README.md` with [specific sections]
- [ ] Create API documentation for new endpoints
- [ ] Add implementation notes in [specific location]

## 🛠️ Implementation Process
1. [First step with details]
2. [Second step with details]
3. [Remaining steps...]

## 🔄 Related Components
**Similar Implementations:** [List relevant examples in codebase]
**Dependent Components:** [List components that will use this implementation]
**Required Libraries:** [List specific libraries with versions]

## ✅ Acceptance Criteria
- [ ] All specified tests pass
- [ ] Code follows project architecture
- [ ] Real-world testing validates functionality
- [ ] Documentation is complete and accurate
- [ ] Code meets quality standards (specify tools/metrics)

## ✓ Implementation Validation
**Code Quality Checks:** [List specific quality checks]
**Functional Verification:** [List verification steps]
**Documentation Verification:** [List documentation requirements]
```

## Common Pitfalls to Avoid

### ❌ Vague Requirements
**Poor:** "Implement data storage for NCAA information."
**Better:** "Create a single DuckDB database file at `data/ncaa.duckdb` with the schema defined in the Technical Requirements section."

### ❌ Unspecific Testing Instructions
**Poor:** "Write tests for the implementation."
**Better:** "Implement the following test cases in `tests/test_ncaa_data.py`:
- `test_fetch_data_success`: Verifies successful API data retrieval
- `test_store_data_formatting`: Ensures data is properly formatted before storage"

### ❌ Ambiguous Architecture Guidance
**Poor:** "Follow the project's database architecture."
**Better:** "Use a single DuckDB database file at `data/ncaa.duckdb`. Do NOT create separate database files for different data types. All tables should be created in this single file."

### ❌ Missing Verification Steps
**Poor:** "Make sure it works."
**Better:** "Execute `python -m ncaa.scripts.fetch --start-date 2023-01-01 --end-date 2023-01-31` and verify that:
1. Data is retrieved without errors
2. The database file contains the expected number of records
3. The console output matches the expected format"

### ❌ Overly Detailed Implementation
**Poor:** Providing 50+ lines of code with all error handling and edge cases.
**Better:** Providing a 10-15 line example that illustrates the pattern with key interfaces.

### ❌ Missing Conversation Context
**Poor:** Creating a task with no reference to decisions made in prior discussions.
**Better:** "Based on our discussion on [date], we decided to use JWT for authentication because of [specific reasons]. This task implements that approach."

### ❌ Assuming Context Persistence
**Poor:** Referring to "the approach we discussed" without specifying what that approach was.
**Better:** Explicitly stating "Implement the repository pattern using the Active Record approach with these specific methods: [list methods]"

## Implementation Verification Checklist

Before considering a task complete, verify that:

1. **Tests:**
   - [ ] All specified tests are implemented
   - [ ] Tests run before implementation (RED)
   - [ ] Implementation passes all tests (GREEN)
   - [ ] Code is refactored while maintaining passing tests (REFACTOR)
   - [ ] Real-world testing with actual data confirms functionality

2. **Architecture:**
   - [ ] Implementation follows specified architecture
   - [ ] File paths and naming conventions match requirements
   - [ ] Database schema matches specifications
   - [ ] No architecture anti-patterns are introduced

3. **Documentation:**
   - [ ] All required documentation is created/updated
   - [ ] Implementation decisions are documented
   - [ ] API documentation is complete
   - [ ] Code includes appropriate comments

4. **Verification:**
   - [ ] All verification steps are executed
   - [ ] Results match expected outcomes
   - [ ] Edge cases are tested
   - [ ] Performance meets requirements

## Conclusion

Writing effective tasks for AI coding agents requires clarity, specificity, and thoroughness. By following these guidelines and using the provided template, you can ensure that your AI assistant has all the information needed to successfully implement your requirements.

Remember that AI agents work best with explicit instructions that leave little room for interpretation. Taking the time to craft comprehensive task descriptions will save development time, reduce misunderstandings, and result in higher-quality implementations.
