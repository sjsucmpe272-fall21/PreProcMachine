#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "numpy==1.21.4",
    "pandas==1.3.4",
    "scipy==1.7.3",
    "scikit-learn==1.0.1",
]

test_requirements = []

setup(
    author="William Su, Yash Modi, Niranjan Reddy Masapeta",
    author_email="modiyash3393@gmail.com",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="A tool to automate preprocessing phase.",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="preprocmachine",
    name="preprocmachine",
    packages=find_packages(exclude=["tests"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/yashm28sjsu/preprocmachine",
    version="0.0.2",
    zip_safe=False,
)
