from setuptools import setup, find_packages

setup(
    name="conllu-pos-dataset",
    version="0.1.0",
    description="A dataset processor for CoNLL-U formatted PoS tagging tasks using Hugging Face datasets.",
    author="Mazouz Abderahim",
    author_email="mazouzceminfo@gmail.com",
    url="https://github.com/aspirant2018/conllu-pos-dataset",  # Replace with your actual repo URL
    packages=find_packages(exclude=["scripts", "conllu_u", "tests"]),
    include_package_data=True,
    install_requires=[
        "pandas",
        "scikit-learn",
        "datasets",
        "conllu",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Or whatever license you choose
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
