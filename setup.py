import os
from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    install_requires = fh.read().splitlines()

version = "0.1.0"

with open(os.path.join("src", "version.py"), "w") as f:
    f.writelines([
        '"""This file is auto-generated by setup.py, please do not alter."""\n',
        f'__version__ = "{version}"\n',
        "",
    ])


setup(
    name="src",
    version=version,
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="thanhtvt",
    author_email="trantrongthanhhp@gmail.com",
    packages=find_packages(),
    install_requires=install_requires,
    python_requires=">=3.9",
)