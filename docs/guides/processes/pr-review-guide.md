---
title: AI Agent PR Review Guide
description: Guidelines for AI agents to effectively review pull requests in the NCAA Basketball Analytics project
---

# AI Agent PR Review Guide

## Quick Reference

For effective PR review, evaluate:

- ✅ Functional correctness: Does the implementation work as expected?
- ✅ Code quality: Does the code follow best practices and project standards?
- ✅ Test coverage: Are there sufficient tests that verify correctness?
- ✅ Documentation: Is the documentation complete and accurate?
- ✅ Architecture alignment: Does the implementation follow project architecture?
- ✅ Issue requirements: Does the PR address all requirements from the issue?

## Introduction

This guide provides a structured approach for AI agents to review pull requests in the NCAA Basketball Analytics project. It outlines objective criteria, assessment methodologies, and a decision framework to ensure consistent, thorough reviews that maintain project quality standards.

## PR Review Process

The PR review process follows these steps:

1. **Initial Assessment**: Review PR description, commits, and context
2. **Code Review**: Analyze code changes with objective criteria
3. **Test Verification**: Verify test coverage and correctness
4. **Documentation Review**: Ensure documentation is complete
5. **Requirement Verification**: Check against issue requirements
6. **Decision**: Approve, request changes, or comment with feedback

## Initial Assessment Checklist

Begin every review with this initial assessment:

```markdown
## 🔍 Initial Assessment

- [ ] PR description provides clear overview of changes
- [ ] PR links to related issue(s)
- [ ] PR size is appropriate for effective review
- [ ] PR includes implementation details and decisions
- [ ] PR includes test results and coverage metrics
- [ ] PR includes self-review checklist
```

## Code Review Criteria

### Functional Correctness

```markdown
## 🛠️ Functional Correctness

- [ ] Implementation works as expected for happy paths
- [ ] Implementation correctly handles error cases
- [ ] Implementation is backwards compatible (if applicable)
- [ ] Implementation includes proper validation
- [ ] Implementation performs as expected with representative data
```

### Code Quality

```markdown
## 📊 Code Quality Metrics

- [ ] Code passes linting with defined rules
- [ ] Code passes type checking with no errors
- [ ] Code follows project naming conventions
- [ ] Code avoids duplication
- [ ] Code has appropriate comments for complex logic
- [ ] Functions and methods are appropriately sized
- [ ] Code follows separation of concerns principle
```

### Architecture Alignment

```markdown
## 🏗️ Architecture Alignment

- [ ] Implementation follows project architecture patterns
- [ ] File organization matches project structure
- [ ] Component responsibilities are appropriate
- [ ] Interfaces are well-defined and consistent
- [ ] Dependencies are properly managed
- [ ] Security considerations are addressed
```

## Test Verification Framework

### Test Coverage Analysis

```markdown
## 🧪 Test Verification

- [ ] Unit tests cover all new functionality
- [ ] Integration tests verify component interactions
- [ ] Edge cases are tested appropriately
- [ ] Test coverage meets minimum threshold (>90%)
- [ ] Tests are properly structured and maintainable
- [ ] Manual testing is documented with results
```

### Test Quality Criteria

Evaluate test quality with these criteria:

1. **Isolation**: Tests should be independent of each other
2. **Readability**: Tests should clearly show what's being tested
3. **Completeness**: Tests should cover happy paths, error paths, and edge cases
4. **Performance**: Tests should run quickly to enable rapid feedback
5. **Maintenance**: Tests should be easy to maintain over time

## Documentation Review

```markdown
## 📝 Documentation Review

- [ ] Documentation is updated to reflect changes
- [ ] API documentation is complete and accurate
- [ ] Examples are provided for new functionality
- [ ] Architecture documentation is updated (if applicable)
- [ ] Internal documentation (comments, docstrings) is helpful
- [ ] README and other user-facing docs are updated
```

## Requirement Verification

```markdown
## ✅ Requirement Verification

- [ ] PR addresses all requirements in the related issue(s)
- [ ] Acceptance criteria from the issue are all met
- [ ] Any scope changes are documented and justified
- [ ] Implementation follows the approach outlined in the issue
- [ ] Any deviations from the issue are explained and justified
```

## Decision Framework

Use this framework to make consistent review decisions:

### Approve if:

- All critical review criteria are met (functional correctness, code quality, test coverage, documentation)
- PR addresses all requirements from the issue
- Architecture alignment is maintained
- No security or performance concerns exist

### Request Changes if:

- Critical functionality is missing or incorrectly implemented
- Tests are missing or insufficient
- Documentation is incomplete or inaccurate
- Code quality issues affect maintainability
- Architecture alignment is violated
- Security or performance issues exist

### Comment if:

- Minor improvements could be made but are not required
- Alternative approaches might be worth considering
- Documentation could be enhanced but is sufficient
- Additional tests would be beneficial but coverage is adequate

## Providing Effective Feedback

When requesting changes, ensure feedback is:

1. **Specific**: Point to exact lines and issues
2. **Actionable**: Provide clear guidance on how to fix
3. **Educational**: Explain why the change is needed
4. **Prioritized**: Indicate which changes are critical vs. nice-to-have
5. **Constructive**: Focus on the code, not the author

Example feedback format:
```markdown
## Feedback: [Component/File]

### Critical Issues

1. **Issue**: [Describe specific issue]
   - **Location**: [File path and line numbers]
   - **Problem**: [Why this is an issue]
   - **Solution**: [Specific action to resolve]

2. **Issue**: [Describe specific issue]
   - **Location**: [File path and line numbers]
   - **Problem**: [Why this is an issue]
   - **Solution**: [Specific action to resolve]

### Suggestions for Improvement

1. **Suggestion**: [Describe suggestion]
   - **Location**: [File path and line numbers]
   - **Benefit**: [Why this would improve the code]
   - **Example**: [Example of the suggested approach]
```

## Objective Metrics for Review

Use these objective metrics to ensure consistent reviews:

### Code Quality Metrics

- **Linting Score**: Must pass configured linters with no errors
- **Type Safety**: Must pass type checking with no errors
- **Complexity**: Functions should have reasonable cyclomatic complexity (≤15)
- **Line Length**: Functions should be reasonably sized (≤50 lines preferred)
- **Duplication**: No significant code duplication (>5 lines duplicated)

### Test Metrics

- **Coverage**: >90% line coverage for new code
- **Test Count**: Appropriate number of tests for functionality
- **Test Types**: Appropriate mix of unit, integration, and edge case tests
- **Test Results**: All tests must pass

### Documentation Metrics

- **API Coverage**: 100% of public API documented
- **Example Coverage**: Examples provided for key functionality
- **Architectural Docs**: Updated for significant changes

## Common Review Scenarios

### 1. Missing Tests

```markdown
## Feedback: Missing Tests

### Critical Issues

1. **Issue**: Insufficient test coverage for error handling
   - **Location**: `src/data/repository.py` (lines 45-60)
   - **Problem**: Error handling code isn't covered by tests
   - **Solution**: Add tests for the following error scenarios:
     - Invalid data format
     - Database connection failure
     - Concurrent modification conflicts

2. **Issue**: Missing integration test
   - **Location**: `src/api/endpoints/teams.py`
   - **Problem**: New endpoint needs integration testing
   - **Solution**: Add integration test that verifies the endpoint correctly interacts with the repository layer
```

### 2. Architecture Misalignment

```markdown
## Feedback: Architecture Misalignment

### Critical Issues

1. **Issue**: Direct database access in API layer
   - **Location**: `src/api/endpoints/teams.py` (lines 25-30)
   - **Problem**: API layer should not directly access database; violates separation of concerns
   - **Solution**: Move database operations to the repository layer and have the API layer use the repository

2. **Issue**: Inconsistent error handling pattern
   - **Location**: `src/data/repository.py` (lines 70-85)
   - **Problem**: Error handling doesn't follow project pattern of using custom exception types
   - **Solution**: Use the custom exception types defined in `src/utils/errors.py`
```

### 3. Documentation Issues

```markdown
## Feedback: Documentation Issues

### Critical Issues

1. **Issue**: Missing API documentation
   - **Location**: `src/api/endpoints/teams.py` (lines 15-40)
   - **Problem**: New endpoint lacks API documentation
   - **Solution**: Add docstrings describing parameters, return values, and error responses

2. **Issue**: Outdated architecture documentation
   - **Location**: `docs/architecture/data-flow.md`
   - **Problem**: Diagram doesn't reflect new data flow with the repository layer
   - **Solution**: Update diagram to include the new repository layer
```

## Review Response Protocol

When the PR author responds to your review:

1. **Re-review Changes**: Evaluate the changes made in response to feedback
2. **Track Resolution**: Verify each issue has been addressed
3. **Incremental Approval**: Approve specific changes while leaving other issues open
4. **Final Approval**: Approve the PR once all critical issues are resolved

Example re-review comment:
```markdown
## Re-review

### Resolved Issues
- ✅ Tests now cover error handling scenarios
- ✅ API documentation is complete

### Remaining Issues
- ❌ Architecture misalignment: API layer still directly accesses database
```

## Performance and Security Review

For changes that might affect performance or security:

```markdown
## 🔐 Security Review

- [ ] Authentication and authorization are properly implemented
- [ ] Input validation prevents injection attacks
- [ ] Sensitive data is properly protected
- [ ] Error messages don't expose sensitive information
- [ ] Security best practices are followed

## ⚡ Performance Review

- [ ] Database queries are optimized
- [ ] N+1 query problems are avoided
- [ ] Performance-critical operations are benchmarked
- [ ] Resource usage is reasonable
- [ ] Potential bottlenecks are identified and addressed
```

## PR Approval Template

When approving a PR, use this template:

```markdown
## 👍 Approval

I've reviewed this PR and verified:

- ✅ Code meets all quality standards
- ✅ Tests are comprehensive and passing
- ✅ Documentation is complete and accurate
- ✅ All requirements from the issue are addressed
- ✅ Architecture alignment is maintained

This PR is ready to merge.
```

## PR Change Request Template

When requesting changes, use this template:

```markdown
## 🔄 Change Request

I've reviewed this PR and found some issues that need to be addressed:

### Critical Issues
[List issues with specific locations and solutions]

### Suggestions for Improvement
[List optional improvements]

Please address the critical issues before this PR can be approved.
```

## PR Comment Template

When leaving a comment, use this template:

```markdown
## 💬 Comment

I've reviewed this PR and have some suggestions:

### Suggestions for Improvement
[List optional improvements]

These are optional improvements - feel free to address them or not.
```

## Review Decision Rubric

Use this rubric to ensure consistent decision-making:

```
| Category           | Approve                     | Request Changes            | Comment                    |
|--------------------|-----------------------------|-----------------------------|----------------------------|
| Functional Correctness | All functionality works correctly | Critical functionality issues | Minor edge cases not handled |
| Code Quality       | Passes all quality metrics  | Significant quality issues  | Minor style improvements    |
| Test Coverage      | >90% coverage, all cases    | Missing critical tests      | Could use additional tests  |
| Documentation      | Complete and accurate       | Missing critical docs       | Could use more examples     |
| Architecture       | Follows architecture        | Violates architecture       | Minor structural improvements |
| Requirements       | All requirements met        | Missing required features   | Enhancement suggestions     |
```

## Conclusion

Effective PR reviews are critical to maintaining code quality and project standards. By following this guide, AI agents can provide consistent, thorough, and helpful reviews that improve code quality while supporting rapid development.

Remember that reviews should be focused on code improvement, not just finding issues. Strive to provide constructive feedback that helps the author understand how to improve their code while acknowledging good work and creative solutions.

Focus on making objective assessments based on project standards rather than subjective preferences. Use the decision framework to ensure consistent review outcomes that maintain project quality.
