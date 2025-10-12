"""Credentials module."""

import os
from pathlib import Path
from typing import Dict, Union, cast

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from dotenv import load_dotenv
from snowflake.snowpark import Session


def _load_env_from_locations() -> None:
    """Try loading a .env file from several sensible locations.

    Order of precedence:
      1. SNOWFLAKE_DOTENV_PATH env var (absolute path)
      2. Project root .env (two levels up from this file)
      3. CWD .env
    """
    # 1) Explicit override
    env_override = os.getenv("SNOWFLAKE_DOTENV_PATH")
    if env_override:
        load_dotenv(dotenv_path=env_override)
        return

    # 2) Project root (assumes this file is under src/snowflake_ml_template/...)
    possible_root = Path(__file__).resolve().parents[3]
    root_env = possible_root / ".env"
    if root_env.exists():
        load_dotenv(dotenv_path=str(root_env))
        return

    # 3) CWD fallback
    cwd_env = Path.cwd() / ".env"
    if cwd_env.exists():
        load_dotenv(dotenv_path=str(cwd_env))
        return

    # Last resort: let dotenv attempt default behaviour
    load_dotenv()


_load_env_from_locations()


def get_snowflake_session() -> Session:
    """Create and return a Snowflake Snowpark Session using key pair authentication.

    Expects the following environment variables to be set:
      - SNOWFLAKE_ACCOUNT
      - SNOWFLAKE_USER
      - SNOWFLAKE_WAREHOUSE
      - SNOWFLAKE_DATABASE
      - SNOWFLAKE_SCHEMA
      - SNOWFLAKE_ROLE
      - SNOWFLAKE_PRIVATE_KEY_PATH
      - SNOWFLAKE_PRIVATE_KEY_PASSPHRASE (optional)
    """
    account = os.getenv("SNOWFLAKE_ACCOUNT")
    user = os.getenv("SNOWFLAKE_USER")
    warehouse = os.getenv("SNOWFLAKE_WAREHOUSE")
    database = os.getenv("SNOWFLAKE_DATABASE")
    schema = os.getenv("SNOWFLAKE_SCHEMA")
    role = os.getenv("SNOWFLAKE_ROLE")
    key_path = os.getenv("SNOWFLAKE_PRIVATE_KEY_PATH")
    passphrase = os.getenv("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE", "")
    password = os.getenv("SNOWFLAKE_PASSWORD")

    # Validate core connection params
    missing_core = [
        k
        for k in [
            "SNOWFLAKE_ACCOUNT",
            "SNOWFLAKE_USER",
            "SNOWFLAKE_WAREHOUSE",
            "SNOWFLAKE_DATABASE",
            "SNOWFLAKE_SCHEMA",
            "SNOWFLAKE_ROLE",
        ]
        if not os.getenv(k)
    ]
    if missing_core:
        raise EnvironmentError(f"Missing Snowflake core env vars: {missing_core}")

    # At this point, all core variables are guaranteed to be strings (validated above)
    account = cast(str, account)
    user = cast(str, user)
    warehouse = cast(str, warehouse)
    database = cast(str, database)
    schema = cast(str, schema)
    role = cast(str, role)

    # Prefer key-pair auth if key provided
    connection_parameters: Dict[str, Union[str, bytes]] = {
        "account": account,
        "user": user,
        "warehouse": warehouse,
        "database": database,
        "schema": schema,
        "role": role,
    }

    if key_path:
        # Load private key
        with open(key_path, "rb") as key_file:
            key_data = key_file.read()
            private_key = serialization.load_pem_private_key(
                key_data,
                password=passphrase.encode() if passphrase else None,
                backend=default_backend(),
            )

        # Convert private key to DER bytes for Snowflake
        pkb = private_key.private_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        connection_parameters["private_key"] = pkb
    elif password:
        # Fall back to password auth if available
        connection_parameters["password"] = password
    else:
        raise EnvironmentError(
            "Neither SNOWFLAKE_PRIVATE_KEY_PATH nor SNOWFLAKE_PASSWORD found in environment."
        )

    # Build and return the Snowpark session
    return Session.builder.configs(connection_parameters).create()
