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

```shell
# Install uv if it is not installed
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

