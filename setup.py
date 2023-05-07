import setuptools

setuptools.setup(
    name="vocabtest",
    version="0.1",
    author="Pol van Rijn",
    author_email="pol.van-rijn@ae.mpg.de",
    description="Code to create vocabulary tests for an open ended number of languages",
    description_long="Code to create vocabulary tests for an open ended number of languages. Here we do it for Wikipedia (called WikiVocab) and for the Bible (called BibleVocab).",
    url="https://github.com/polvanrijn/VocabTest",
    packages=["vocabtest"],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
)
