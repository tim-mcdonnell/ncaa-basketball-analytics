from setuptools import setup, find_packages

setup(
    name="ncaa-basketball-analytics",
    version="0.1.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    python_requires=">=3.12",
    install_requires=[
        "aiohttp>=3.11.14",
        "pydantic>=2.7.1",
        "tenacity>=9.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "pytest-asyncio>=0.23.2",
        ],
    },
)
