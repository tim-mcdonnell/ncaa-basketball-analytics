---
title: Configuration Migration Guide
description: How to handle configuration changes and migrations in NCAA Basketball Analytics
---

# Configuration Migration Guide

This guide explains how to handle configuration schema changes and migrations in the NCAA Basketball Analytics project.

## Configuration Versioning

The configuration system uses semantic versioning to track changes to the configuration schema:

```yaml
version: "1.0"  # Format: MAJOR.MINOR
```

Version components have specific meanings:

- **MAJOR**: Incompatible changes that require migration
- **MINOR**: Backward-compatible additions or changes

## Version Compatibility Rules

The configuration system enforces the following compatibility rules:

1. The application requires configuration with a matching MAJOR version
2. The application accepts a MINOR version that is less than or equal to its expected MINOR version
3. Missing version numbers default to "1.0"

## Detecting Version Incompatibility

When loading configuration, the system checks version compatibility:

```python
from ncaa_basketball.config import Config, ConfigVersionError

try:
    config = Config.load()
except ConfigVersionError as e:
    print(f"Configuration version error: {e}")
    print(f"Expected version: {e.expected_version}")
    print(f"Found version: {e.found_version}")
```

## Migration Strategies

### 1. Minor Version Changes (Backward Compatible)

For minor version changes, the configuration system handles the compatibility automatically:

- New optional fields are added with defaults
- Existing fields maintain their original behavior

Example of a minor version change (1.0 → 1.1):

```yaml
# New field added in version 1.1
models:
  inference:
    batch_size: 128
    threshold: 0.5
    cache_results: true  # New optional field in 1.1
```

### 2. Major Version Changes (Breaking Changes)

For major version changes, a migration is required:

1. **Manual Migration**: Update configuration files manually
2. **Automated Migration**: Use the built-in migration utility

#### Using the Migration Utility

The configuration system includes a migration utility:

```python
from ncaa_basketball.config.migration import migrate_config

# Migrate from version 1.0 to 2.0
migrate_config("config/base.yaml", target_version="2.0")
```

This updates the configuration file in place, or optionally creates a new file:

```python
# Create a new file instead of modifying the original
migrate_config("config/base.yaml",
               output_file="config/base.v2.yaml",
               target_version="2.0")
```

## Migration Examples

### Example 1: Field Renamed (Major Version Change)

If a field is renamed in version 2.0:

```yaml
# Version 1.0
database:
  db_host: "localhost"

# Version 2.0
database:
  host: "localhost"  # Renamed from db_host
```

Migration code:

```python
def migrate_v1_to_v2(config_data):
    """Migrate from v1 to v2 configuration schema."""
    if "database" in config_data and "db_host" in config_data["database"]:
        # Copy the value from old field to new field
        config_data["database"]["host"] = config_data["database"]["db_host"]
        # Remove the old field
        del config_data["database"]["db_host"]

    # Update version
    config_data["version"] = "2.0"
    return config_data
```

### Example 2: Structure Change (Major Version Change)

If the structure changes significantly:

```yaml
# Version 1.0
api:
  timeout: 30
  rate_limit: 100

# Version 2.0
api:
  client:
    timeout: 30
  limits:
    requests_per_minute: 100
```

Migration code:

```python
def migrate_v1_to_v2(config_data):
    """Migrate from v1 to v2 configuration schema."""
    if "api" in config_data:
        # Create new structure
        api_config = config_data["api"]
        new_api_config = {
            "client": {"timeout": api_config.get("timeout", 30)},
            "limits": {"requests_per_minute": api_config.get("rate_limit", 100)}
        }
        config_data["api"] = new_api_config

    # Update version
    config_data["version"] = "2.0"
    return config_data
```

## Creating a Custom Migration

To create a custom migration:

1. Create a migration function that transforms the configuration
2. Register the migration with the migration system

```python
from ncaa_basketball.config.migration import register_migration

def my_custom_migration(config_data):
    # Transform configuration data
    # ...
    return transformed_data

# Register migration from version 1.0 to 2.0
register_migration("1.0", "2.0", my_custom_migration)
```

## Best Practices for Configuration Changes

1. **Prefer Minor Versions**: When possible, make backward-compatible changes
2. **Document Changes**: Document all changes in the changelog
3. **Provide Migration Guide**: For major versions, provide a detailed migration guide
4. **Test Migrations**: Verify that migrations work correctly with test cases
5. **Preserve Comments**: The migration utility preserves comments in YAML files
6. **Validate After Migration**: Always validate configuration after migration

## Handling Migration in Production

For production systems:

1. Create a backup of the current configuration
2. Test the migration in a staging environment
3. Apply the migration during a maintenance window
4. Verify that the application works correctly with the migrated configuration

## Troubleshooting Migrations

If you encounter issues during migration:

- Check the migration logs for detailed error messages
- Verify that the source configuration is valid before migration
- Manually inspect the migrated configuration file
- Run validation on the migrated configuration
