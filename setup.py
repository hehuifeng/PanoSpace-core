from __future__ import annotations

from itertools import chain
from pathlib import Path

from setuptools import find_packages, setup

PACKAGE_NAME = "panospace"
REPO_URL = "https://github.com/hehuifeng/PanoSpace"
BASE_DIR = Path(__file__).parent.resolve()
README = (BASE_DIR / "README.md").read_text(encoding="utf8")

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------
# Keep a single source of truth for the version.  ``panospace._version`` uses
# importlib.metadata to determine the version at runtime, but during ``setup``
# we explicitly set it here to avoid the "0+unknown" fallback.
VERSION = "0.1.0"

# ---------------------------------------------------------------------------
# Dependency groups
# ---------------------------------------------------------------------------
BASE_REQUIREMENTS = [
    "numpy>=1.23",
    "pandas>=1.5",
    "anndata>=0.10",
    "scanpy>=1.9",
    "scipy>=1.9",
    "tqdm>=4.65",
    "Pillow>=9.4",
    "opencv-python>=4.8",
    "matplotlib>=3.7",
    "scikit-image>=0.22",
    "requests>=2.31",
    "ray>=2.7",
    "pot>=0.9",
    "qpsolvers>=4.0",
]

CELLVIT_REQUIREMENTS = [
    "torch>=2.0",
    "torchvision>=0.15",
]

ANNOTATION_REQUIREMENTS = [
    "pyro-ppl>=1.8",
    "pytorch-lightning>=2.1",
    "lightning>=2.1",
    "transformers>=4.33",
    "gurobipy>=10",
]

PREDICTION_REQUIREMENTS: list[str] = []  # scipy already covered above

EXTRAS_REQUIRE = {
    "cellvit": sorted(set(BASE_REQUIREMENTS + CELLVIT_REQUIREMENTS)),
    "annotation": sorted(set(BASE_REQUIREMENTS + ANNOTATION_REQUIREMENTS)),
    "prediction": sorted(set(BASE_REQUIREMENTS + PREDICTION_REQUIREMENTS)),
}

ALL_OPTIONAL = sorted({
    requirement
    for requirement in chain.from_iterable(EXTRAS_REQUIRE.values())
})
EXTRAS_REQUIRE["all"] = ALL_OPTIONAL
EXTRAS_REQUIRE["dev"] = sorted({
    "black>=24.3",
    "pre-commit>=3.5",
    "pytest>=7.4",
    "pytest-cov>=4.1",
})

INSTALL_REQUIRES = ALL_OPTIONAL

# ---------------------------------------------------------------------------
# Setup metadata
# ---------------------------------------------------------------------------
setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description="High-resolution spatial transcriptomics analysis toolkit.",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Hui-Feng He",
    author_email="huifeng@mails.ccnu.edu.cn",
    url=REPO_URL,
    project_urls={
        "Documentation": REPO_URL,
        "Source": REPO_URL,
        "Tracker": f"{REPO_URL}/issues",
    },
    license="MIT",
    packages=find_packages(exclude=("tests", "tests.*")),
    include_package_data=True,
    package_data={
        "panospace._core.annotation._RCTD_backend": [
            "extdata/*.txt",
            "extdata/*.txt.gz",
        ],
    },
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    keywords=[
        "spatial transcriptomics",
        "single-cell",
        "deep learning",
        "bioinformatics",
    ],
)
