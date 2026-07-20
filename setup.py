from setuptools import setup, find_packages

setup(
    name="flashrank-pro",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.38.0",
        "sentence-transformers>=3.0.0",
        "datasets>=2.14.0",
        "accelerate>=0.25.0",
        "openai>=1.0.0",
        "numpy",
        "tqdm",
        "fire",
    ],
    python_requires=">=3.9",
)
