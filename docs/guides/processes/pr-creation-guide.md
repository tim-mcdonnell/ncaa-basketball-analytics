---
title: AI Agent PR Creation Guide
description: Guidelines for AI agents to create effective and reviewable pull requests in the NCAA Basketball Analytics project
---

# AI Agent PR Creation Guide

## Quick Reference

For efficient PR creation, ensure each PR includes:

- ✅ Clear title referencing issue(s) with the format: `[Issue #X] Brief description`
- ✅ Comprehensive description with implementation details and decisions
- ✅ Pre-submission validation results (tests, linting, type checking)
- ✅ Link to the original GitHub issue
- ✅ Self-review checklist with all items verified
- ✅ Documentation updates related to changes

## Introduction

This guide helps AI agents create high-quality, reviewable pull requests that align with project standards and streamline the review process. Following these guidelines ensures that PRs are consistently structured, thoroughly documented, and properly validated before review.

## PR Preparation Checklist

Before creating a PR, ensure:

```markdown
## 🔍 Pre-PR Checklist

- [ ] All tests are passing (unit, integration, e2e)
- [ ] Code passes all linting rules
- [ ] Type checking passes with no errors
- [ ] Documentation is updated to reflect changes
- [ ] All acceptance criteria from the issue are met
- [ ] Implementation follows architectural guidelines
- [ ] No debug code or commented-out code exists
- [ ] Sensitive information is not exposed
```

## PR Structure

### Title Format

Use the format: `[Issue #X] Brief description of the change`

Examples:
- `[Issue #42] Implement DuckDB repository pattern for team data`
- `[Issue #15] Fix race condition in API data fetching`
- `[Issue #23, #24] Refactor authentication system`

### Description Template

```markdown
## 🔄 Changes Overview

This PR implements [brief description of the change].

## 📋 Issue Reference

Resolves #[issue number]

## 🛠️ Implementation Details

[Detailed description of the implementation approach]

### Key Components

- **Component 1**: [Description of component]
- **Component 2**: [Description of component]

### Technical Decisions

1. **Decision**: [Technical decision made]
   - **Rationale**: [Why this decision was made]
   - **Alternatives**: [Alternatives considered]

2. **Decision**: [Technical decision made]
   - **Rationale**: [Why this decision was made]
   - **Alternatives**: [Alternatives considered]

## 🧪 Testing Details

**Unit Tests**: [Description of unit tests added/modified]
**Integration Tests**: [Description of integration tests added/modified]
**Manual Testing**: [Description of manual testing performed]

## 📊 Test Results

```
[Include test output here]
```

## 🔍 Self-Review Checklist

- [ ] Implementation meets all acceptance criteria
- [ ] Code follows project architecture guidelines
- [ ] All tests are passing
- [ ] Code passes linting and type checking
- [ ] Documentation is updated
- [ ] No TODOs or commented-out code remains
- [ ] No debug code or print statements remain
- [ ] Implementation handles error cases
- [ ] Performance considerations addressed

## 📝 Documentation Updates

- [List documentation files updated]
- [Summary of documentation changes]

## 🚀 Deployment Considerations

- [List any deployment steps or considerations]
- [Database migrations, environment variables, etc.]
```

## Commit Guidelines

Structure commits logically with:

1. **Atomic Changes**: Each commit should represent a single logical change
2. **Descriptive Messages**: Use the format `[Component] Short description of change`
3. **Issue References**: Include issue numbers in commit messages when relevant
4. **Consistent Scope**: Group related changes in the same commit

Example commit messages:
```
[DB] Create DuckDB schema for teams table
[API] Implement team repository pattern
[Tests] Add team repository unit tests
[Docs] Update data storage documentation
```

## Implementation Details Section

The Implementation Details section should include:

1. **Architectural Overview**: How the changes fit into the project architecture
2. **Key Components**: The main components modified or created
3. **Technical Decisions**: Important decisions made during implementation
4. **Alternatives Considered**: Other approaches that were considered but rejected
5. **Future Considerations**: Any follow-up work or future improvements

Example:
```markdown
## 🛠️ Implementation Details

This PR implements a DuckDB-based repository pattern for team data storage.

### Key Components

- **DB Connection Manager**: Handles database connection lifecycle
- **Team Repository**: Implements CRUD operations for team data
- **Schema Validation**: Validates team data before storage

### Technical Decisions

1. **Decision**: Used connection pooling for database access
   - **Rationale**: Improves performance for concurrent operations
   - **Alternatives**: Single connection approach was considered but rejected due to potential concurrency issues

2. **Decision**: Implemented optimistic locking for updates
   - **Rationale**: Prevents data corruption in concurrent update scenarios
   - **Alternatives**: Pessimistic locking was considered but would reduce concurrency
```

## Testing Details Section

Document testing thoroughly with:

1. **Test Coverage**: Summary of what aspects are tested
2. **Test Types**: Unit, integration, e2e, manual testing details
3. **Edge Cases**: How edge cases are addressed
4. **Test Results**: Actual results from test runs
5. **Performance Metrics**: If applicable

Example:
```markdown
## 🧪 Testing Details

**Unit Tests**: Added 15 unit tests covering:
- Repository CRUD operations
- Data validation
- Error handling

**Integration Tests**: Added 3 integration tests covering:
- Database connection and transactions
- Repository integration with API layer

**Manual Testing**: Verified with sample data:
- Team creation with valid/invalid data
- Team update scenarios
- Concurrent operations

## 📊 Test Results

```
Running 18 tests...
✓ test_create_team_success (0.02s)
✓ test_create_team_invalid_data (0.01s)
...
✓ test_concurrent_updates (0.05s)

18 tests passed, 0 failed
Coverage: 94%
```
```

## Self-Review Process

The AI agent should perform a thorough self-review before submitting the PR:

1. **Functional Verification**: Verify that all functionality works as expected
2. **Code Quality**: Check code against project standards
3. **Test Completeness**: Ensure test coverage is adequate
4. **Documentation**: Verify documentation is complete and accurate
5. **Issue Requirements**: Check that all issue requirements are met

The self-review checklist should be included in the PR description with all items checked.

## Documentation Updates

Document all changes to project documentation:

1. **User-Facing Docs**: Changes to user-facing documentation
2. **Developer Docs**: Changes to developer documentation
3. **Architecture Docs**: Changes to architecture documentation
4. **Code Comments**: Updates to code comments and docstrings

Example:
```markdown
## 📝 Documentation Updates

- Updated `docs/architecture/data-storage.md` with the new repository pattern
- Added docstrings to all public functions in the repository classes
- Updated `README.md` with database setup instructions
- Added examples in `docs/examples/team-data-access.md`
```

## Deployment Considerations

Include any deployment-specific information:

1. **Database Migrations**: Any database changes required
2. **Environment Variables**: New or changed environment variables
3. **Dependencies**: New external dependencies
4. **Backwards Compatibility**: Any backwards compatibility considerations
5. **Rollback Plan**: How to roll back if issues occur

Example:
```markdown
## 🚀 Deployment Considerations

- Database Migration: Run `python manage.py migrate_teams`
- New Environment Variables:
  - `DUCKDB_PATH`: Path to DuckDB database file
- Dependencies: Added duckdb-engine==0.9.2 to requirements.txt
- Backward Compatibility: This change is backward compatible with existing data
```

## Common PR Issues to Avoid

### ❌ Vague PR Descriptions
**Poor:** "Implemented team data storage."
**Better:** "Implemented DuckDB-based repository pattern for team data with CRUD operations, validation, and comprehensive test coverage."

### ❌ Missing Implementation Decisions
**Poor:** "Used DuckDB for the database."
**Better:** "Selected DuckDB over SQLite because of its analytical capabilities, in-memory performance, and compatibility with our data processing pipeline."

### ❌ Incomplete Test Results
**Poor:** "Tests are passing."
**Better:** Include actual test output showing all tests passing with coverage metrics.

### ❌ Ignored Self-Review Items
**Poor:** Checkboxes marked without verification.
**Better:** Actually verify each item and provide evidence where applicable.

### ❌ Missing Related Documentation
**Poor:** Code changes without documentation updates.
**Better:** Update all relevant documentation alongside code changes.

## PR Size Guidelines

Keep PRs manageable for effective review:

1. **Small PRs (Preferred)**: <300 lines of code, focused on a single feature or bug fix
2. **Medium PRs**: 300-500 lines of code, may include multiple related components
3. **Large PRs (Avoid)**: >500 lines of code, should be split if possible

If a task requires a large PR, consider:
- Breaking it into smaller, sequential PRs
- Creating a draft PR early for preliminary feedback
- Providing additional context and organization in the PR description

## Review Readiness Checklist

Before marking a PR as ready for review, verify:

```markdown
## ✅ Review Readiness

- [ ] PR addresses all requirements in the linked issue(s)
- [ ] PR includes comprehensive tests with >90% coverage
- [ ] PR passes all CI checks (tests, linting, type checking)
- [ ] PR includes updated documentation
- [ ] PR has been self-reviewed with all items addressed
- [ ] PR includes evidence of manual testing
- [ ] PR is appropriately sized for effective review
```

## Conclusion

Creating effective pull requests is critical to maintaining code quality and streamlining the review process. By following these guidelines, AI agents can create consistent, reviewable PRs that include all necessary information for reviewers to understand, evaluate, and approve changes efficiently.

Remember that the ultimate goal of a PR is not just to merge code, but to ensure that changes are well-understood, correctly implemented, thoroughly tested, and properly documented.
