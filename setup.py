import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("version", "r") as fh:
    version = fh.read()

setuptools.setup(
    # Here is the module name.
    name="libreasr",
    # version of the module
    version=version,
    # Name of Author
    author="Christian Bruckdorfer",
    # your Email address
    author_email="christiansvde@freenet.de",
    # Specifying that we are using markdown file for description
    long_description=long_description,
    long_description_content_type="text/markdown",
    # Any link to reach this module, ***if*** you have any webpage or github profile
    url="https://github.com/iceychris/LibreASR",
    packages=setuptools.find_packages(),
    #     install_requires=[
    #      "package1",
    #    "package2",
    #    ],
    license="MIT",
    # classifiers like program is suitable for python3, just leave as it is.
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
