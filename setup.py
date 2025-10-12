"""Legacy packaging shim for editable installs during development.

This file enables `python setup.py develop` as a fallback for editable installs
when PEP 660 editable metadata preparation fails on some environments.
"""

from pathlib import Path

from setuptools import find_packages, setup

HERE = Path(__file__).parent
README = (
    (HERE / "README.md").read_text(encoding="utf8")
    if (HERE / "README.md").exists()
    else ""
)


setup(
    name="snowflake-ml-template",
    version="0.1.0",
    description="A comprehensive template for building scalable ML pipelines on Snowflake",
    long_description=README,
    long_description_content_type="text/markdown",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    zip_safe=False,
)
