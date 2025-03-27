---
title: Configuration Guide
description: A guide to using and customizing the configuration system in NCAA Basketball Analytics
---

# Configuration Guide

This guide explains how to use the configuration system in the NCAA Basketball Analytics project.

## Overview

The NCAA Basketball Analytics project uses a hierarchical, environment-aware configuration system based on YAML files. The configuration system provides:

- Type validation through Pydantic models
- Environment-specific configuration (development, testing, production)
- Override capability through environment variables
- Version compatibility checking
- Dot notation access for convenient property retrieval

## Base Configuration Structure

The base configuration consists of several components:

```yaml
version: "1.0"  # Configuration schema version

api:
  base_url: "https://api.example.com"
  timeout: 30
  rate_limit: 100

database:
  host: "localhost"
  port: 5432
  name: "ncaa_basketball"
  user: "user"
  password: ""  # Set through environment variables for security

features:
  compute_threads: 4
  cache_enabled: true
  storage_path: "./data/features"

models:
  training:
    batch_size: 64
    epochs: 100
    learning_rate: 0.001
  inference:
    batch_size: 128
    threshold: 0.5

dashboard:
  theme: "light"
  update_interval: 60
  cache_timeout: 300
```

## Using the Configuration System

The configuration system is designed to be easy to use:

```python
from ncaa_basketball.config import Config

# Load configuration (automatically uses the current environment)
config = Config.load()

# Access configuration values using dot notation
api_url = config.api.base_url
db_host = config.database.host
batch_size = config.models.training.batch_size

# Or using dictionary-style access
api_url = config["api"]["base_url"]
```

## Configuration Files

Configuration files are stored in the `config/` directory with the following structure:

- `config/base.yaml`: Base configuration with default values
- `config/development.yaml`: Development environment overrides
- `config/testing.yaml`: Testing environment overrides
- `config/production.yaml`: Production environment overrides

When using the configuration system, values are loaded from the base configuration first, then overridden by environment-specific values if available.

## Environment Variables

For sensitive information or deployment-specific settings, you can use environment variables to override configuration values. The system looks for environment variables with the prefix `NCAA_BASKETBALL_`, followed by the configuration path in uppercase with underscores.

Examples:
- `NCAA_BASKETBALL_DATABASE_PASSWORD` overrides `database.password`
- `NCAA_BASKETBALL_API_KEY` overrides `api.key`
- `NCAA_BASKETBALL_FEATURES_STORAGE_PATH` overrides `features.storage_path`

## Custom Configuration

To extend or customize the configuration system:

1. Update the Pydantic models in `ncaa_basketball/config/models.py`
2. Update the base configuration in `config/base.yaml`
3. Create or update environment-specific overrides in the appropriate YAML files

## Validation

The configuration system validates all settings against the defined Pydantic models. This ensures that:

- Required fields are present
- Values have the correct types
- Constraints are enforced (min/max values, string patterns, etc.)

If validation fails, the system will raise a detailed error message indicating what's wrong.

## Best Practices

1. **Keep sensitive data out of configuration files** - Use environment variables for passwords, API keys, etc.
2. **Use the dot notation access** - It's more readable and provides better IDE support
3. **Set reasonable defaults** - Ensure the base configuration works without modifications for development
4. **Document configuration changes** - When adding new configuration options, update this guide
5. **Use version numbers** - When making breaking changes to the configuration schema, update the version number
