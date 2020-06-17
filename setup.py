import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mocaplib",
    version="0.0.2",
    author="Moon Ki Jung",
    author_email="m.k.jung@outlook.com",
    description="Library for Motion Capture data processing and analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mkjung99/mocaplib",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)