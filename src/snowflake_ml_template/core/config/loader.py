"""Configuration loader with support for multiple sources.

This module provides a flexible configuration loader that can load and merge
configurations from multiple sources:
- YAML files
- Environment variables
- Python dictionaries
- Pydantic models

It supports hierarchical configuration with defaults, environment-specific
overrides, and environment variable substitution.

Classes:
    ConfigLoader: Load and merge configurations from multiple sources
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar, Union

import yaml
from pydantic import BaseModel, ValidationError

from snowflake_ml_template.utils.logging import StructuredLogger, get_logger

T = TypeVar("T", bound=BaseModel)


class ConfigLoader:
    """Load and merge configurations from multiple sources.

    This class provides a flexible configuration loading system that supports:
    - Loading from YAML files
    - Loading from environment variables
    - Merging multiple configurations with priority
    - Environment variable substitution in values
    - Type-safe configuration using Pydantic models

    Configuration priority (highest to lowest):
    1. Explicit overrides passed to load()
    2. Environment variables
    3. Environment-specific config file (e.g., config/prod.yaml)
    4. Default config file (e.g., config/defaults.yaml)

    Attributes:
        config_dir: Directory containing configuration files
        environment: Current environment (dev, test, prod)
        _logger: Logger instance

    Example:
        >>> loader = ConfigLoader(config_dir="config", environment="dev")
        >>>
        >>> # Load Snowflake configuration
        >>> snowflake_config = loader.load(
        ...     SnowflakeConfig,
        ...     config_file="snowflake.yaml"
        ... )
        >>>
        >>> # Load with overrides
        >>> config = loader.load(
        ...     SnowflakeConfig,
        ...     config_file="snowflake.yaml",
        ...     overrides={"warehouse": "ML_TRAINING_WH"}
        ... )
        >>>
        >>> # Load from environment variables
        >>> config = loader.load_from_env(
        ...     SnowflakeConfig,
        ...     prefix="SNOWFLAKE_"
        ... )
    """

    def __init__(
        self, config_dir: Union[str, Path] = "config", environment: str = "dev"
    ) -> None:
        """Initialize the configuration loader.

        Args:
            config_dir: Directory containing configuration files
            environment: Current environment (dev, test, prod)

        Raises:
            ValueError: If environment is invalid
        """
        if environment not in ["dev", "test", "prod"]:
            raise ValueError(
                f"Invalid environment: {environment}. "
                "Must be one of: dev, test, prod"
            )

        self.config_dir = Path(config_dir)
        self.environment = environment
        self._logger = self._get_logger()

        self._logger.info(
            f"Initialized ConfigLoader: dir={self.config_dir}, env={environment}"
        )

    def _get_logger(self) -> StructuredLogger:
        """Return a structured logger scoped to the loader."""
        return get_logger(__name__)

    def load(
        self,
        model_class: Type[T],
        config_file: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
        use_env_vars: bool = True,
        env_prefix: str = "",
    ) -> T:
        """Load configuration from multiple sources.

        This method loads configuration with the following priority:
        1. Explicit overrides
        2. Environment variables (if use_env_vars=True)
        3. Environment-specific config file
        4. Default config file

        Args:
            model_class: Pydantic model class to instantiate
            config_file: Optional config file name (relative to config_dir)
            overrides: Optional dictionary of override values
            use_env_vars: Whether to load from environment variables
            env_prefix: Prefix for environment variables (e.g., "SNOWFLAKE_")

        Returns:
            Instantiated and validated Pydantic model

        Raises:
            ValidationError: If configuration is invalid
            FileNotFoundError: If config file doesn't exist

        Example:
            >>> config = loader.load(
            ...     SnowflakeConfig,
            ...     config_file="snowflake.yaml",
            ...     overrides={"warehouse": "ML_TRAINING_WH"}
            ... )
        """
        self._logger.info(f"Loading configuration for {model_class.__name__}")

        # Start with empty config
        merged_config: Dict[str, Any] = {}

        # 1. Load from default config file (if exists)
        if config_file:
            default_path = self.config_dir / "defaults" / config_file
            if default_path.exists():
                self._logger.debug(f"Loading defaults from {default_path}")
                default_config = self._load_yaml(default_path)
                merged_config = self._deep_merge(merged_config, default_config)

        # 2. Load from environment-specific config file (if exists)
        if config_file:
            env_path = self.config_dir / "environments" / f"{self.environment}.yaml"
            if env_path.exists():
                self._logger.debug(f"Loading environment config from {env_path}")
                env_config = self._load_yaml(env_path)

                # Extract section for this config if it exists
                config_name = config_file.replace(".yaml", "").replace(".yml", "")
                if config_name in env_config:
                    merged_config = self._deep_merge(
                        merged_config, env_config[config_name]
                    )
                else:
                    merged_config = self._deep_merge(merged_config, env_config)

        # 3. Load from environment variables (if enabled)
        if use_env_vars:
            self._logger.debug(
                f"Loading from environment variables with prefix: {env_prefix}"
            )
            env_config = self._load_from_env(model_class, env_prefix)
            merged_config = self._deep_merge(merged_config, env_config)

        # 4. Apply explicit overrides
        if overrides:
            self._logger.debug(f"Applying overrides: {list(overrides.keys())}")
            merged_config = self._deep_merge(merged_config, overrides)

        # 5. Substitute environment variables in values
        merged_config = self._substitute_env_vars(merged_config)

        # 6. Instantiate and validate Pydantic model
        try:
            config: T = model_class(**merged_config)
            self._logger.info(f"Successfully loaded {model_class.__name__}")
            return config
        except ValidationError as e:
            self._logger.error(
                f"Configuration validation failed for {model_class.__name__}: {e}"
            )
            raise

    def load_from_env(self, model_class: Type[T], prefix: str = "") -> T:
        """Load configuration from environment variables only.

        Args:
            model_class: Pydantic model class to instantiate
            prefix: Prefix for environment variables

        Returns:
            Instantiated and validated Pydantic model

        Example:
            >>> # With environment variables:
            >>> # SNOWFLAKE_ACCOUNT=my_account
            >>> # SNOWFLAKE_USER=my_user
            >>> # SNOWFLAKE_WAREHOUSE=ML_TRAINING_WH
            >>> config = loader.load_from_env(
            ...     SnowflakeConfig,
            ...     prefix="SNOWFLAKE_"
            ... )
        """
        env_config = self._load_from_env(model_class, prefix)
        config: T = model_class(**env_config)
        return config

    def load_from_dict(self, model_class: Type[T], config_dict: Dict[str, Any]) -> T:
        """Load configuration from a dictionary.

        Args:
            model_class: Pydantic model class to instantiate
            config_dict: Configuration dictionary

        Returns:
            Instantiated and validated Pydantic model

        Example:
            >>> config_dict = {
            ...     "account": "my_account",
            ...     "user": "my_user",
            ...     "warehouse": "ML_TRAINING_WH"
            ... }
            >>> config = loader.load_from_dict(SnowflakeConfig, config_dict)
        """
        if not isinstance(config_dict, dict):
            raise ValueError("config_dict must be a dictionary")
        config: T = model_class.model_validate(config_dict)
        return config

    def _load_yaml(self, file_path: Path) -> Dict[str, Any]:
        """Load configuration from a YAML file.

        Args:
            file_path: Path to YAML file

        Returns:
            Configuration dictionary

        Raises:
            FileNotFoundError: If file doesn't exist
            yaml.YAMLError: If YAML is invalid
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")

        try:
            with open(file_path, "r") as f:
                config = yaml.safe_load(f)
                if config is None:
                    return {}
                if not isinstance(config, dict):
                    raise ValueError(
                        f"Expected YAML to parse to a dictionary, got {type(config).__name__}"
                    )
                return config
        except yaml.YAMLError as e:
            self._logger.error(f"Failed to parse YAML file {file_path}: {e}")
            raise

    def _load_from_env(self, model_class: Type[T], prefix: str = "") -> Dict[str, Any]:
        """Load configuration from environment variables.

        This method looks for environment variables matching the field names
        of the Pydantic model, optionally with a prefix.

        Args:
            model_class: Pydantic model class
            prefix: Prefix for environment variables

        Returns:
            Configuration dictionary
        """
        config: Dict[str, Any] = {}

        # Get field names from Pydantic model
        for field_name in model_class.model_fields.keys():
            env_var_name = f"{prefix}{field_name.upper()}"
            env_value = os.environ.get(env_var_name)

            if env_value is not None:
                # Try to parse as appropriate type
                config[field_name] = self._parse_env_value(env_value)

        return config

    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value to appropriate type.

        Args:
            value: String value from environment variable

        Returns:
            Parsed value (str, int, float, bool, or dict)
        """
        # Try to parse as boolean
        if value.lower() in ["true", "yes", "1"]:
            return True
        if value.lower() in ["false", "no", "0"]:
            return False

        # Try to parse as integer
        try:
            return int(value)
        except ValueError:
            pass

        # Try to parse as float
        try:
            return float(value)
        except ValueError:
            pass

        # Try to parse as JSON (for dicts/lists)
        if value.startswith("{") or value.startswith("["):
            try:
                import json

                return json.loads(value)
            except json.JSONDecodeError:
                pass

        # Return as string
        return value

    def _deep_merge(
        self, base: Dict[str, Any], override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Deep merge two dictionaries.

        This method recursively merges override into base, with override
        values taking precedence.

        Args:
            base: Base dictionary
            override: Override dictionary

        Returns:
            Merged dictionary
        """
        result = base.copy()

        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                # Recursively merge nested dictionaries
                result[key] = self._deep_merge(result[key], value)
            else:
                # Override value
                result[key] = value

        return result

    def _substitute_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Substitute environment variables in configuration values.

        This method looks for values in the format ${VAR_NAME} and replaces
        them with the corresponding environment variable value.

        Args:
            config: Configuration dictionary

        Returns:
            Configuration with substituted values
        """
        result: Dict[str, Any] = {}

        for key, value in config.items():
            new_value: Any
            if isinstance(value, dict):
                # Recursively substitute in nested dictionaries
                new_value = self._substitute_env_vars(value)
            elif (
                isinstance(value, str)
                and value.startswith("${")
                and value.endswith("}")
            ):
                # Substitute environment variable
                env_var_name = value[2:-1]
                env_value = os.environ.get(env_var_name)
                if env_value is not None:
                    new_value = env_value
                else:
                    self._logger.warning(
                        f"Environment variable {env_var_name} not found, "
                        f"keeping original value: {value}"
                    )
                    new_value = value
            else:
                new_value = value

            result[key] = new_value

        return result

    def save_to_yaml(self, config: BaseModel, file_path: Union[str, Path]) -> None:
        """Save configuration to a YAML file.

        Args:
            config: Pydantic model instance
            file_path: Path to save YAML file

        Example:
            >>> config = SnowflakeConfig(...)
            >>> loader.save_to_yaml(config, "config/snowflake.yaml")
        """
        file_path = Path(file_path)

        # Create directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert Pydantic model to dict
        config_dict = config.model_dump()

        # Write to YAML file
        with open(file_path, "w") as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)

        self._logger.info(f"Saved configuration to {file_path}")
