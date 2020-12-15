import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="autoMMM", # Replace with your own username
    version="0.0.1",
    author="Nick Gustafson",
    author_email="nick_gustafson@outlook.com",
    description="Used for easliy implementing Marketing Mix Models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NichoGustafson/autoMMM",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
