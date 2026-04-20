from setuptools import setup, find_packages

setup(
    name="domcs_eeg",
    version="1.0.0",
    description=(
        "DOMCS-EEG: Security-Aware Cross-State EEG Biometric Verification "
        "via Protocol Realism and Representation Disentanglement"
    ),
    author="Kanimozhi L, S. Shridevi",
    author_email="kanimozhi.l2024@vitstudent.ac.in",
    url="https://github.com/KanimozhiL16/eeg-biometric-disentangle",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0",
        "numpy>=1.21",
        "scikit-learn>=1.0",
        "scipy>=1.7",
        "pyyaml>=6.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
