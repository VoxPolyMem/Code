# VoxPolyMem

VoxPolyMem is a multimodal long-term memory framework for multi-party dialogue. It organizes dialogue evidence into linked memory representations and retrieves relevant context for grounded question answering. The repository provides the method implementation and evaluation entry points for Mem-Gallery, H2HMem, and VoxPolyBench.

## Getting started

Python 3.12 is required. Create an environment outside the repository, then install the dependencies:

```bash
git clone https://github.com/VoxPolyMem/Code.git
cd Code
python3.12 -m venv ../.venv-voxpolymem
source ../.venv-voxpolymem/bin/activate
python -m pip install -r requirements.txt
```

Copy the example configuration and set your model endpoint, credentials, embedding service, and dataset paths:

```bash
cp .env.example .env
# Edit .env for your environment.
set -a
source ./.env
set +a
export PYTHON_BIN="$(command -v python)"
```

The launch scripts use these exported settings. Source `.env` again in each new shell. The default experimental settings are already defined in the provided launch scripts; users only need to configure environment-dependent paths and services. Benchmark data and model weights are not redistributed here; see [data requirements](data/README.md).

## Evaluation

Build memory before running question answering:

```bash
bash scripts/build_memgallery_memory_v1.sh
bash scripts/run_memgallery_v1.sh

bash scripts/build_h2hmem_memory_v1.sh
bash scripts/run_h2hmem_v1.sh
```

For VoxPolyBench, follow the [audio evaluation guide](docs/VOXPOLYBENCH_REPRODUCTION.md). Evaluation requires the corresponding dataset and configured model services.

The offline verification check does not make model calls:

```bash
python scripts/verify_release.py
```

## Repository structure

- `core/`: memory organization and retrieval components.
- `adapters/`: benchmark input and memory construction adapters.
- `evaluation/`: question answering and evaluation entry points.
- `audio/`: speaker identity components.
- `experiments/`: VoxPolyBench integration.
- `scripts/`: launch and verification scripts.

Implementation details and the scope of result reproduction are documented in the [reproducibility notes](docs/REPRODUCIBILITY.md).
