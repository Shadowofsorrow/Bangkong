from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="bangkong-llm",
    version="0.1.0",
    author="Soni Nugraha",
    author_email="bilbobangkong@gmail.com",
    description="A production-ready, cloud-native system for training and deploying large language models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/bangkong",
    project_urls={
        "Bug Tracker": "https://github.com/your-org/bangkong/issues",
        "Documentation": "https://github.com/your-org/bangkong/docs",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "bangkong-train=scripts.train:main",
            "bangkong-eval=scripts.evaluate:main",
            "bangkong-convert=scripts.convert:main",
            "bangkong-deploy=scripts.deploy:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
