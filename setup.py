from setuptools import setup, find_packages

setup(
    name="ezbedrock",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "boto3>=1.28.0",
        "pydantic>=2.0.0",
    ],
    author="Mohamed Alturfi",
    description="A simple wrapper for AWS Bedrock API",
    url="https://github.com/MohamedAlturfiSVG/ezbedrock"
)