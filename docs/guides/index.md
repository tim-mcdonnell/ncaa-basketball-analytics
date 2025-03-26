# Developer Guides

This section provides practical guides for working with the NCAA Basketball Analytics project. These guides are designed to help developers understand how to perform common tasks and implement new features.

## Available Guides

- [Getting Started](getting-started.md): Set up your development environment and run the project
- [Adding Features](adding-features.md): Guide to implementing new features in the system
- [Training Models](training-models.md): Guide to training and evaluating prediction models

## Common Tasks

### Development Setup

To set up a development environment, follow the [Getting Started](getting-started.md) guide which covers:

- Repository setup
- Dependencies installation
- Configuration
- Running tests

### Feature Development

When implementing new features:

1. First understand the requirements and how they fit into the system
2. Follow the [Test-Driven Development](getting-started.md#test-driven-development) approach
3. Write tests that validate the feature behavior
4. Implement the feature code
5. Submit a pull request with comprehensive documentation

### Data Pipeline Debugging

When debugging data pipeline issues:

1. Check the logs in the appropriate directory based on component
2. Use the Airflow UI to examine task status and logs
3. Verify configuration settings
4. Check for data quality issues in the raw inputs

For more detailed guidance, refer to the specific guides in this section. 