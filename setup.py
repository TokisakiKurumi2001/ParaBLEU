import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='ReWord',
    packages=['ReWord'],
    version='0.1.0',
    description='Reorder word in English sentence to follow correct grammar',
    author='Le Minh Khoi',
    license='MIT',
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    py_modules=["ReWord"],
    install_requires=['torch', 'pytorch-lightning', 'wandb', 'transformers', 'datasets', 'tokenizers', 'sacrebleu', 'sacremoses']
)