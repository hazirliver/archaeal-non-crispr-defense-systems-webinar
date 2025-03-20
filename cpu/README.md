# CPU Environment Setup

## Prerequisites

- Git
- CMake (3.0 or higher)
- C++ compiler with C++11 support (GCC 4.8+, Clang 3.8+, or equivalent)
- CPU with AVX2 support for optimal performance

## Installation

### 1. MMseqs2 Installation

[MMseqs2](https://github.com/soedinglab/MMseqs2) is a software suite for ultra-fast protein sequence searching and clustering.

```shell
# Clone the MMseqs2 repository
git clone https://github.com/soedinglab/MMseqs2.git
cd MMseqs2

# Create and navigate to the build directory
mkdir build
cd build

# Configure the build with AVX2 support
cmake -DCMAKE_BUILD_TYPE=RELEASE \
      -DCMAKE_INSTALL_PREFIX=. \
      -DHAVE_AVX2=1 \
      ..

# Compile and install MMseqs2
make -j$(nproc)
make install

# Add MMseqs2 to the system PATH
export PATH=$(pwd)/bin/:$PATH
```

For permanent PATH modification, add the export command to your `.bashrc` or `.zshrc` file:

```shell
echo 'export PATH=/path/to/MMseqs2/build/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

### 2. Python Environment Setup

Setup a Python environment with [Marimo](https://marimo.io) (interactive notebooks), [ESM](https://github.com/facebookresearch/esm) (protein language models), and other data science tools:

```shell
# Install uv if it is not installed (faster Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
cd ../../marimo/

# Create and activate a Python virtual environment
uv venv -p 3.11 --seed
source .venv/bin/activate

# Install necessary Python packages
uv pip install marimo polars biobear matplotlib numpy esm==3.1.2 torch umap-learn

# Launch Marimo in headless mode
uv run marimo edit --headless --port 12123 --host 0.0.0.0
```

## Usage

After installation, you can access the Marimo notebook interface by navigating to:
```
http://localhost:12123
```
