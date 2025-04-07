from setuptools import find_packages, setup

with open("requirements.txt") as f:
    required = f.read().splitlines()


setup(
    name="src",
    packages=find_packages()
)
