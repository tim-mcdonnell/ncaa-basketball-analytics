---
title: Environment Configuration Guide
description: How to use environment-specific configuration in NCAA Basketball Analytics
---

# Environment Configuration Guide

The NCAA Basketball Analytics project supports different environments (development, testing, production) with environment-specific configuration settings.

## Available Environments

The system supports the following environments:

- **Development**: For local development work (default)
- **Testing**: For automated tests and CI/CD pipelines
- **Production**: For deployed production instances

## Environment Selection

The current environment is determined by:

1. The `NCAA_BASKETBALL_ENV` environment variable (if set)
2. Falling back to "development" if not specified

Example of setting the environment:

```bash
# Linux/macOS
export NCAA_BASKETBALL_ENV=production

# Windows (Command Prompt)
set NCAA_BASKETBALL_ENV=production

# Windows (PowerShell)
$env:NCAA_BASKETBALL_ENV="production"
```

## Environment-Specific Configuration Files

Each environment has its own configuration file that overrides values from the base configuration:

- `config/development.yaml` - Development environment settings
- `config/testing.yaml` - Testing environment settings
- `config/production.yaml` - Production environment settings

Example `development.yaml`:

```yaml
# Development environment configuration
database:
  host: "localhost"

features:
  cache_enabled: true

models:
  training:
    epochs: 10  # Reduced for faster development cycles
```

Example `production.yaml`:

```yaml
# Production environment configuration
api:
  timeout: 60  # Increased timeout for production

database:
  host: "db.production.example.com"

features:
  compute_threads: 16  # More compute resources in production

dashboard:
  cache_timeout: 600  # Longer cache in production
```

## Configuration Loading Process

When loading the configuration, the system:

1. Loads the base configuration (`config/base.yaml`)
2. Identifies the current environment
3. Loads and applies overrides from the environment-specific file
4. Applies any environment variable overrides
5. Validates the final configuration

## Accessing Environment in Code

You can check the current environment in your code:

```python
from ncaa_basketball.config import Config, Environment

config = Config.load()

# Check environment
if config.environment == Environment.DEVELOPMENT:
    # Development-specific logic
    print("Running in development mode")
elif config.environment == Environment.PRODUCTION:
    # Production-specific logic
    print("Running in production mode")
```

## Testing with Different Environments

For testing with different environments:

```python
# Using context manager to temporarily change environment
from ncaa_basketball.config import Config, using_environment

# This will load the production configuration within the context
with using_environment("production"):
    config = Config.load()
    # Test with production settings

# Back to default environment outside the context
config = Config.load()  # Default environment
```

## Best Practices

1. **Keep environments consistent**: Ensure all environments have the same configuration structure with different values
2. **Minimize differences**: Only override values that need to be different between environments
3. **Test all environments**: Verify your code works correctly in all target environments
4. **Document environment requirements**: List any specific requirements for each environment
5. **Use feature flags**: For features that should be enabled/disabled per environment, use feature flags in the configuration
