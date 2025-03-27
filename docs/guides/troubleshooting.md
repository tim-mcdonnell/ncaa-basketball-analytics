---
title: Configuration Troubleshooting Guide
description: Common issues and solutions when working with the NCAA Basketball Analytics configuration system
---

# Configuration Troubleshooting Guide

This guide helps diagnose and resolve common issues with the configuration system in NCAA Basketball Analytics.

## Common Issues and Solutions

### Configuration Not Loading

**Issue**: The configuration fails to load with a file not found error.

**Possible causes**:
- Configuration file doesn't exist in the expected location
- Incorrect permissions on the configuration file
- Invalid path provided to the configuration loader

**Solutions**:
1. Verify the file exists: `ls -la config/base.yaml`
2. Check file permissions: `chmod 644 config/base.yaml`
3. Use absolute paths or correct relative paths:
   ```python
   import os
   config_path = os.path.join(os.path.dirname(__file__), "config", "base.yaml")
   config = Config.load(config_dir=os.path.dirname(config_path))
   ```

### Validation Errors

**Issue**: Configuration fails to validate with Pydantic errors.

**Possible causes**:
- Missing required fields
- Incorrect data types
- Constraint violations (min/max values, string patterns)

**Solutions**:
1. Check the error messages for the specific issue
2. Review the required fields in the Pydantic models
3. Correct the values in your configuration file:
   ```yaml
   # Fix type error: api.timeout should be an integer
   api:
     timeout: 30  # Not "30"
   ```

### Environment-Specific Configuration Not Applied

**Issue**: Environment-specific overrides are not being applied.

**Possible causes**:
- Environment not set correctly
- Environment-specific file not found
- Incorrect override format

**Solutions**:
1. Verify the environment is set: `echo $NCAA_BASKETBALL_ENV`
2. Check the environment-specific file exists:
   ```bash
   ls -la config/production.yaml  # If using production environment
   ```
3. Ensure overrides use the correct structure:
   ```yaml
   # Must match the base configuration structure
   database:
     host: "production-db.example.com"
   ```

### Version Compatibility Errors

**Issue**: Version compatibility error when loading configuration.

**Possible causes**:
- Major version mismatch between configuration and application
- Configuration file has no version field

**Solutions**:
1. Check the expected version in the error message
2. Update the configuration file to match the expected major version:
   ```yaml
   version: "2.0"  # Update to match expected version
   ```
3. Use the migration utility to migrate configuration:
   ```python
   from ncaa_basketball.config.migration import migrate_config
   migrate_config("config/base.yaml", target_version="2.0")
   ```

### Environment Variable Overrides Not Working

**Issue**: Environment variables are not overriding configuration values.

**Possible causes**:
- Incorrect environment variable naming
- Type conversion issues
- Variable not set in the environment

**Solutions**:
1. Use the correct prefix and format:
   ```bash
   # Format: NCAA_BASKETBALL_SECTION_KEY
   export NCAA_BASKETBALL_DATABASE_HOST="production-db.example.com"
   ```
2. Verify the variable is set: `env | grep NCAA_BASKETBALL`
3. Ensure the type is compatible (strings for string values, numbers for numeric values)

### Dot Notation Access Errors

**Issue**: Dot notation access to configuration throws attribute errors.

**Possible causes**:
- Trying to access a non-existent property
- Configuration not loaded properly

**Solutions**:
1. Check if the property exists in the configuration:
   ```python
   # Use .get() with default for potentially missing values
   value = config.get("missing.property", default="default_value")
   ```
2. Ensure the configuration is loaded correctly before access
3. Use dictionary-style access as an alternative:
   ```python
   # Dictionary-style access with get() for safety
   value = config["section"].get("property", "default")
   ```

### Nested Configuration Access Issues

**Issue**: Difficulty accessing deeply nested configuration values.

**Possible causes**:
- Complex configuration structure
- Inconsistent access patterns

**Solutions**:
1. Use the path-based access method:
   ```python
   # Access deeply nested value with a path string
   value = config.get("models.training.hyperparameters.learning_rate")
   ```
2. Break down the access into steps with null checks:
   ```python
   models = config.models
   if models and hasattr(models, "training"):
       training = models.training
       if hasattr(training, "hyperparameters"):
           learning_rate = training.hyperparameters.learning_rate
   ```

## Debugging Techniques

### Enable Debug Logging

To see more detailed information about configuration loading:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("ncaa_basketball.config")
```

### Inspect Raw Configuration

To inspect the raw configuration data before validation:

```python
from ncaa_basketball.config.loaders import load_yaml_file

# Load raw data without validation
raw_config = load_yaml_file("config/base.yaml")
print(raw_config)
```

### Validate Specific Sections

To validate just one section of the configuration:

```python
from ncaa_basketball.config.models import ApiConfig
from pydantic import ValidationError

api_data = {"base_url": "https://api.example.com", "timeout": "30"}  # Note: timeout should be an int

try:
    api_config = ApiConfig(**api_data)
    print("Valid configuration:", api_config)
except ValidationError as e:
    print("Validation errors:", e)
```

### Check Environment Variables

To see all configuration-related environment variables:

```python
import os
env_vars = {k: v for k, v in os.environ.items() if k.startswith("NCAA_BASKETBALL_")}
print("Configuration environment variables:", env_vars)
```

## Advanced Troubleshooting

### Configuration Loading Process

If you need to diagnose issues with the configuration loading process:

```python
from ncaa_basketball.config.core import Config
from ncaa_basketball.config.environment import get_current_environment

# Check current environment
env = get_current_environment()
print(f"Current environment: {env}")

# Load base configuration explicitly
config = Config.load(env="development", config_dir="config")
print(f"Loaded configuration for {config.environment}")
```

### Test Configuration Roundtrip

To test if your configuration can be loaded and saved correctly:

```python
import tempfile
import os
from ncaa_basketball.config import Config
from ncaa_basketball.config.loaders import save_yaml_file

config = Config.load()

# Save to a temporary file
with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
    save_yaml_file(config.to_dict(), tmp.name)
    tmp_path = tmp.name

# Try to load it back
try:
    reloaded_config = Config.load(config_dir=os.path.dirname(tmp_path),
                                 config_file=os.path.basename(tmp_path))
    print("Configuration roundtrip successful!")
except Exception as e:
    print(f"Configuration roundtrip failed: {e}")
finally:
    os.unlink(tmp_path)  # Clean up
```

## Getting Help

If you're still experiencing issues:

1. Check the configuration system unit tests for examples of correct usage
2. Review the documentation in the `docs/architecture/config-management.md` file
3. Search for similar issues in the project's issue tracker
4. Create a minimal reproduction of the issue for easier debugging
