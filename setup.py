import setuptools

# with open("README.md", "r") as fh:
#     long_description = fh.read()

setuptools.setup(
    name="spkit",
    version="0.0.2",
    author="Laura Moraes",
    author_email="lmoraes@cos.ufrj.br",
    description="Package to build different models for predicting student performance",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
