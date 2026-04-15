from setuptools import setup, find_packages

setup(
    name="text_rich_mllm",
    version="0.1.0",
    description="A unified text-rich multimodal reasoning pipeline for documents, charts, and scientific figures.",
    author="Project Team",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.40.0",
        "datasets>=2.16.0",
        "peft>=0.7.0",
        "accelerate>=0.25.0",
        "pyyaml>=6.0",
        "Pillow>=10.0.0"
    ],
)
