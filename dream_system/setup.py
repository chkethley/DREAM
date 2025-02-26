"""Setup configuration for the DREAM system package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="dream-system",
    version="1.0.0",
    author="Craig",
    author_email="craig@example.com",
    description="A Dynamic Response and Engagement Artificial Mind System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/craig/dream-system",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.0",
        "faiss-cpu>=1.7.0",
        "sentence-transformers>=2.0.0",
        "transformers>=4.5.0",
        "aiohttp>=3.8.0",
        "tenacity>=8.0.0",
        "tomlkit>=0.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-asyncio>=0.15.0",
            "black>=21.0",
            "isort>=5.0",
            "mypy>=0.900",
            "flake8>=3.9.0",
        ],
    },
) 