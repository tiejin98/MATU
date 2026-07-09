# MATU: Multi-Agent Tensor Uncertainty

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyPI](https://img.shields.io/pypi/v/matu-uq.svg)](https://pypi.org/project/matu-uq/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/tiejin98/MATU/actions/workflows/ci.yml/badge.svg)](https://github.com/tiejin98/MATU/actions/workflows/ci.yml)

MATU quantifies uncertainty for LLM-based multi-agent systems from repeated
conversation trajectories. This repository supports the ACL 2026 main paper
**"Every Response Counts: Quantifying Uncertainty of LLM-based Multi-Agent
Systems through Tensor Decomposition."**

MATU is log-first: the core input is a conversation log JSON following
[`data/LOG_FORMAT.md`](data/LOG_FORMAT.md). The included generation scripts are
examples, not required infrastructure.

Paper: **Every Response Counts: Quantifying Uncertainty of LLM-based
Multi-Agent Systems through Tensor Decomposition** (ACL 2026 main)

## Dependencies

Core dependencies are specified in [`pyproject.toml`](pyproject.toml),
[`requirements.txt`](requirements.txt), and [`environment.yml`](environment.yml).

```text
numpy>=1.24
tqdm>=4.66
tensorly>=0.8
sentence-transformers>=2.6
transformers>=4.40,<5
accelerate>=0.30
torch>=2.1
scikit-learn>=1.3
datasets>=2.18
PyYAML>=6.0
```

Optional log-generation examples:

```text
openai>=1.0
camel-ai>=0.2.0
```

## Installation

For package use:

```bash
pip install matu-uq
matu --help
matu-uq --help
```

For the full quick start, clone the repository because the sample logs,
reference scores, and embedding archives live in `quick_start/`:

```bash
git clone https://github.com/tiejin98/MATU.git
cd MATU
pip install -e .
```

For development and tests:

```bash
pip install -e ".[dev]"
```

For the optional CAMEL/OpenAI example:

```bash
pip install -e ".[examples]"
```

Conda users can create the same core environment:

```bash
conda env create -f environment.yml
conda activate matu-uq
```

Local secrets and machine-specific paths should live in `.env` files, never in
source:

```bash
cp .env.example .env
cp quick_start/.env.example quick_start/.env
```

OpenAI credentials are only needed for the optional CAMEL/GPT log-generation
example. The included quick-start evaluation does not require an API key.

## Architecture

```text
Conversation logs
      |
      v
Role-wise embeddings
      |
      v
Run / role / step tensor
      |
      v
CP-2 / PARAFAC2 decomposition
      |
      v
MATU uncertainty score
      |
      v
AUROC / AUARC evaluation
```

Optional baselines, such as EigV, start from the same conversation logs and are
evaluated with the same labels. More details are in
[`docs/architecture.md`](docs/architecture.md).

## Quick Start

The [`quick_start/`](quick_start/) folder provides pre-computed sample artifacts
so users can evaluate MATU immediately, inspect the intermediate files, or
re-run selected stages. It covers two public settings:

| Setting | Dataset | Agent Framework / Log Source | LLM | Included Artifacts |
| --- | --- | --- | --- | --- |
| MATH + Camel + Qwen2.5 | MATH | Included Qwen2.5 conversation logs Using Camel. | Qwen2.5-7B-Instruct | Conversation logs, MATH labels, Qwen3 role embeddings, MATU fit curves, scalar uncertainty, and SAUP-Multiple baseline scores. |
| MMLU + AutoGen + Qwen2.5 | MMLU | Included AutoGen multi-agent logs with analyst, verifier, and star roles. | Qwen2.5-7B-Instruct | Conversation logs, MMLU labels, Qwen3 role embeddings, MATU fit curves, and SAUP-Multiple baseline scores. |

All final and intermediate result files needed for the quick-start evaluations are already included. Re-embedding logs and re-running CP-2 are optional debugging/reproduction steps, not required for seeing the reported metrics.

Raw embedding matrices are stored inside zip archives rather than committed as
standalone `.pkl` files. The quick-start scripts can stage the provided MATH
embeddings automatically when needed.

### Quick Start Files

| File | Description |
| --- | --- |
| `quick_start/data/conversation_logs_Math_qwen2.5.json` | MATH repeated conversation logs for the public quick-start sample. |
| `quick_start/data/conversation_logs_MMLU_Autogen_qwen2.5.json` | MMLU AutoGen conversation log sample for the quick-start setting. |
| `quick_start/data/embeddings_Math_qwen2.5_qwen3.zip` | Zipped Qwen3 role embedding matrices for the MATH sample. Extract before inspecting or reusing the raw matrices. |
| `quick_start/data/embeddings_MMLU_Autogen_qwen2.5.zip` | Zipped AutoGen analyst, verifier, and star embedding matrices for the MMLU sample. Extract before inspecting or reusing the raw matrices. |
| `quick_start/results/fit_dict_Math_qwen2.5_qwen3embedding.pkl` | Included MATU fit curves for MATH. |
| `quick_start/results/uncertainty_Math_qwen2.5.pkl` | Included scalar MATU uncertainty for MATH. |
| `quick_start/results/accuracy_dict_Math_qwen2.5.pkl` | MATH repeated-run correctness labels. |
| `quick_start/results/fit_dict_MMLU_Autogen_qwen2.5.pkl` | Included MATU fit curves for the MMLU AutoGen quick-start setting. |
| `quick_start/results/accuracy_dict_MMLU_Autogen_qwen2.5.pkl` | MMLU AutoGen repeated-run correctness labels. |
| `quick_start/results/saup_scores_Math_qwen2.5.pkl` | Included MATH SAUP-Multiple baseline scores for comparison. |
| `quick_start/results/saup_scores_MMLU_Autogen_qwen2.5.pkl` | Included MMLU AutoGen SAUP-Multiple baseline scores for comparison. |

### Option A: Standalone Quick-Start Scripts

Run commands from the repository root. The included reference evaluation reads files already stored in `quick_start/results/`, so it does not require API keys, GPUs, or model downloads. Create the quick-start environment file if you want to run the optional `00` and `01` scripts to read local settings consistently:
`OPENAI_API_KEY` is only needed for OpenAI/CAMEL log generation,
`MATU_MODEL_CACHE` can point to a local Hugging Face cache or model path, and
`MATU_MAX_RANK` controls CP-2 wrapper scripts used for Step `02`.

```bash
cp quick_start/.env.example quick_start/.env
```

Step `00` is optional log generation. The public logs are already included in
`quick_start/data/`, so users only run these scripts when they want to create
new conversation logs.

**Important:** `00_generate_logs_camel_gpt.py` requires `OPENAI_API_KEY` in
`quick_start/.env` and the example dependencies from `pip install -e ".[examples]"`.
`00_generate_logs_hf_qwen.py` requires local or downloadable
Hugging Face model access through `MATU_QWEN_MODEL` and, if needed,
`MATU_MODEL_CACHE`. **GPU is required**.

**You may skip this optional `00` step if you do not have API_key or GPU**

```bash
# 00: MATH + CAMEL/OpenAI example -> new logs under quick_start/generated/
python quick_start/code/00_generate_logs_camel_gpt.py

# 00: MATH + HF/Qwen example -> new logs under quick_start/generated/
python quick_start/code/00_generate_logs_hf_qwen.py
```

Step `01` prepares embedding matrices under
`quick_start/generated/embeddings/`. By default, it stages the provided MATH
embeddings without downloading a model. Set `MATU_RUN_EXTERNAL=1` or
`MATU_MODEL_CACHE` if you want to recompute Qwen3 embeddings from the included
MATH conversation log.

**Important:** Recomputing Step `01` loads `Qwen/Qwen3-Embedding-0.6B`. Make
sure set `MATU_MODEL_CACHE` in `quick_start/.env` if you want to use a local
Hugging Face cache/model path.

```bash
# 01: provided MATH + Qwen3 embeddings -> staged Step 01 outputs
python quick_start/code/01_embed_reference_logs.py
```

The provided Step `01` embedding outputs are stored as zip archives so the
repository does not carry large raw embedding `.pkl` files. When Step `02`
stages the MATH archive with `--embedding-source provided`, or when you extract
the archive manually, the role files are:

- User-role embeddings: `quick_start/generated/reference_embeddings/math/user_embedding_matrices_Math_qwen2.5_qwen3.pkl`
- Assistant-role embeddings: `quick_start/generated/reference_embeddings/math/assistant_embedding_matrices_Math_qwen2.5_qwen3.pkl`

The MMLU AutoGen archive provides one embedding-matrix file per AutoGen role:

- Analyst-role embeddings: `quick_start/generated/reference_embeddings/mmlu_autogen_qwen/autogen_analyst_embedding_matrices_MMLU_HF_qwen2.5.pkl`
- Verifier-role embeddings: `quick_start/generated/reference_embeddings/mmlu_autogen_qwen/autogen_verifier_embedding_matrices_MMLU_HF_qwen2.5.pkl`
- Star-role embeddings: `quick_start/generated/reference_embeddings/mmlu_autogen_qwen/autogen_star_embedding_matrices_MMLU_HF_qwen2.5.pkl`

Step `02` reruns CP-2 through the quick-start wrapper. It can read our provided
MATH embeddings, Step `01` generated embeddings, or explicit embedding pickle
paths. It writes regenerated outputs under `quick_start/generated/results/`;
the included reference fit curves remain in `quick_start/results/`.

```bash
# 02: provided MATH + Qwen3 embeddings -> regenerated MATU fit curves
python quick_start/code/02_run_cp2_from_generated_embeddings.py \
  --embedding-source provided \
  --max_rank 50
```

If you recomputed or staged Step `01`, use the generated source. To pass your
own embedding files, use `--embeddings` with one pickle per role.

```bash
# 02: MATH + generated Qwen3 embeddings -> regenerated MATU fit curves
python quick_start/code/02_run_cp2_from_generated_embeddings.py \
  --embedding-source generated

# 02: custom role embeddings -> regenerated MATU fit curves
python quick_start/code/02_run_cp2_from_generated_embeddings.py \
  --embeddings path/to/user_embedding_matrices.pkl path/to/assistant_embedding_matrices.pkl
```

Step `03` converts fit curves into scalar uncertainty. The reference script
rebuilds the included MATH uncertainty file in `quick_start/results/`; the
generated script converts the optional CP-2 output in
`quick_start/generated/results/`.

```bash
# 03: included MATH fit_dict -> included scalar uncertainty
python quick_start/code/03_fit_to_uncertainty_reference.py

# 03: regenerated MATH fit_dict -> regenerated scalar uncertainty
python quick_start/code/03_fit_to_uncertainty_generated.py
```

Step `04` evaluates MATU. The reference evaluator reads the included result
files for MATH and MMLU AutoGen; the generated evaluator checks the optional
regenerated MATH pipeline.

```bash
# 04: included MATH + Qwen2.5 MATU result
python quick_start/code/04_evaluate_reference_results.py --sample math-qwen

# 04: included MMLU + AutoGen + Qwen2.5 MATU result
python quick_start/code/04_evaluate_reference_results.py --sample mmlu-autogen-qwen

# 04: both included MATU results
python quick_start/code/04_evaluate_reference_results.py --sample all

# 04: regenerated MATH MATU result from optional Steps 01-03
python quick_start/code/04_evaluate_generated_results.py
```

Step `05` evaluates the included SAUP-Multiple baselines for both quick-start
settings.

```bash
# 05: included MATH + Qwen2.5 SAUP-Multiple baseline
python quick_start/code/05_evaluate_baselines.py --sample math-qwen

# 05: included MMLU + AutoGen + Qwen2.5 SAUP-Multiple baseline
python quick_start/code/05_evaluate_baselines.py --sample mmlu-autogen-qwen

# 05: both included SAUP-Multiple baselines
python quick_start/code/05_evaluate_baselines.py --sample all
```

### Option B: Using the MATU CLI

Install from source:

```bash
pip install -e .
```

Run the same core stages through the CLI:

```bash
# MATH + Qwen2.5 logs -> Qwen3 role embeddings
matu embed \
  --logs quick_start/data/conversation_logs_Math_qwen2.5.json \
  --out_dir quick_start/generated/embeddings \
  --roles user assistant

# MATH + generated Qwen3 embeddings -> CP-2 fit curves
matu cp2 \
  --embeddings \
  quick_start/generated/embeddings/user_embedding_matrices.pkl \
  quick_start/generated/embeddings/assistant_embedding_matrices.pkl \
  --out quick_start/generated/results/matu_scores.pkl \
  --legacy_fit_out quick_start/generated/results/fit_dict_generated.pkl \
  --max_rank 50

# MATH + CP-2 fit curves -> scalar MATU uncertainty
matu fit \
  --fit_dict quick_start/generated/results/matu_scores.pkl \
  --out quick_start/generated/results/uncertainty_generated.pkl

# MATH + generated uncertainty -> AUROC/AUARC against included labels
matu eval \
  --uncertainty quick_start/generated/results/uncertainty_generated.pkl \
  --labels quick_start/results/accuracy_dict_Math_qwen2.5.pkl \
  --score_mode raw
```

### Expected Results

The included `.pkl` files reproduce the following quick-start metrics:

| Setting | Method | Source | AUROC | AUARC | Command |
| --- | --- | --- | --- | --- | --- |
| MATH + Camel + Qwen2.5 | MATU | Paper Table 1 | 0.7089 | 0.9064 | Paper reference. |
| MATH + Camel + Qwen2.5 | MATU | Included quick-start artifact | 0.7205 | 0.9017 | `python quick_start/code/04_evaluate_reference_results.py --sample math-qwen` |
| MATH + Camel + Qwen2.5 | SAUP-Multiple | Included quick-start baseline | 0.6078 | 0.8722 | `python quick_start/code/05_evaluate_baselines.py --sample math-qwen` |
| MMLU + AutoGen + Qwen2.5 | MATU | Paper Table 2 | 0.7315 | 0.8833 | Paper reference. |
| MMLU + AutoGen + Qwen2.5 | MATU | Included quick-start artifact | 0.7315 | 0.8834 | `python quick_start/code/04_evaluate_reference_results.py --sample mmlu-autogen-qwen` |
| MMLU + AutoGen + Qwen2.5 | SAUP-Multiple | Included quick-start baseline | 0.7194 | 0.8589 | `python quick_start/code/05_evaluate_baselines.py --sample mmlu-autogen-qwen` |

The MATH artifact was re-run for public release packaging, so it is close to
but not bit-for-bit identical to the paper table. The MMLU AutoGen artifact is
the original artifact used for the paper result, so it matches Table 2 up to
display rounding.

## Full Pipeline

To run MATU on your own conversation logs:

```bash
# 1. Embed role-specific conversation trajectories.
matu embed --config configs/default.yaml \
  --logs path/to/conversation_logs.json \
  --out_dir outputs/embeddings \
  --roles user assistant

# 2. Run CP-2 / PARAFAC2 tensor scoring.
matu cp2 --config configs/default.yaml \
  --embeddings outputs/embeddings/user_embedding_matrices.pkl outputs/embeddings/assistant_embedding_matrices.pkl \
  --out outputs/matu_scores.pkl \
  --legacy_fit_out outputs/fit_dict.pkl

# 3. Convert fit curves to scalar uncertainty.
matu fit --config configs/default.yaml \
  --fit_dict outputs/matu_scores.pkl \
  --out outputs/uncertainty.pkl

# 4. Evaluate uncertainty against repeated-run labels.
matu eval --config configs/default.yaml \
  --uncertainty outputs/uncertainty.pkl \
  --labels path/to/accuracy_dict.pkl
```

Optional baseline:

```bash
matu eigv \
  --logs path/to/conversation_logs.json \
  --mode final \
  --out outputs/eigv_final.pkl
```

## Pipeline Stages

| Stage | Command | Description | Output |
| --- | --- | --- | --- |
| 1. Log collection | `examples/generate_logs_hf_qwen.py` or your own agent framework | Collect repeated multi-agent conversation trajectories. | Conversation log JSON |
| 2. Embedding | `matu embed` | Convert each role's turns into trajectory matrices. | `<role>_embedding_matrices.pkl` |
| 3. CP-2 / MATU | `matu cp2` | Run rank-wise tensor decomposition over repeated trajectories. | Structured MATU scores and optional legacy `fit_dict` |
| 4. Uncertainty conversion | `matu fit` | Convert rank-wise fit curves into scalar uncertainty. | `uncertainty.pkl` |
| 5. Evaluation | `matu eval` | Compute AUROC and AUARC from repeated-run labels. | Console metrics |
| Baseline | `matu eigv` | Compute EigV agreement baseline from logs. | Baseline score pickle |

Key uncertainty definition for legacy `fit_dict` files:

```text
uncertainty = sum_R (1 - fit_R)
```

## Output Directory Structure

```text
outputs/
|-- conversation_logs.json
|-- embeddings/
|   |-- user_embedding_matrices.pkl
|   `-- assistant_embedding_matrices.pkl
|-- matu_scores.pkl
|-- fit_dict.pkl
|-- uncertainty.pkl
`-- eigv_final.pkl

quick_start/generated/
|-- embeddings/
|-- reference_embeddings/
`-- results/
```

Generated outputs, extracted embeddings, model caches, and build artifacts are
ignored by git.

## Configuration

All default paths and hyperparameters are documented in
[`configs/default.yaml`](configs/default.yaml). CLI flags override config file
values.

| Parameter | Default | Description |
| --- | --- | --- |
| `embedding.model` | `Qwen/Qwen3-Embedding-0.6B` | Sentence-transformer embedding model. |
| `embedding.roles` | `[user, assistant]` | Roles to extract from each conversation turn. |
| `cp2.min_rank` | `1` | Minimum CP-2 rank. |
| `cp2.max_rank` | `50` | Maximum CP-2 rank. Reduce for smoke tests. |
| `cp2.max_iter` | `25` | ALS iterations per rank. |
| `cp2.seed` | `0` | Factor initialization seed. |
| `cp2.combine_mode` | `interleave` | How role/run matrices are assembled. |
| `evaluation.error_rule` | `any_incorrect` | Repeated-run error event for AUROC. |

## Data Sources

The quick start is self-contained and does not require benchmark downloads.
Full paper-scale experiments use public datasets including MATH, MMLU,
MoreHopQA, and HumanEval/EvalPlus. Source links and download snippets are in
[`data/README.md`](data/README.md).

## Hardware Requirements

| Stage | Hardware | Estimated Time |
| --- | --- | --- |
| Direct quick-start evaluation | CPU only | Seconds |
| Step 01 provided embedding unzip | CPU only | Seconds |
| Re-embedding sample logs | GPU recommended, CPU possible | Minutes, depending on hardware |
| CP-2 on quick-start embeddings | CPU possible, GPU not required | Minutes to longer for rank 50 |
| Optional CAMEL/GPT log generation | CPU plus OpenAI API key | API dependent |
| Full paper-scale runs | GPU recommended | Dataset and agent-framework dependent |

For a smoke test, reduce `MATU_MAX_RANK` in `quick_start/.env` or
`cp2.max_rank` in `configs/default.yaml`.

## Tests And Makefile

Common checks are collected in the [`Makefile`](Makefile):

```bash
make install-dev
make test
make quick-eval
make paper-eval
make check
make clean
```

`make test` runs the unit tests in `tests/`. `make check` runs tests,
`compileall`, CLI help, MATH quick evaluation, and the MMLU AutoGen
quick-start evaluation. `make clean` removes Python caches, build outputs, and
`:Zone.Identifier` sidecar files.

## Docker

Build a CPU-ready image:

```bash
docker build -t matu-uq .
```

Run quick-start evaluation in the container:

```bash
docker run --rm matu-uq make quick-eval
docker run --rm matu-uq make paper-eval
```

For local outputs:

```bash
docker run --rm -v "$(pwd)/outputs:/app/outputs" matu-uq matu --help
```

## PyPI Package

MATU is available on PyPI:

```bash
pip install matu-uq
```

## Project Structure

```text
MATU/
|-- matu/                  # Main package and CLI
|-- baselines/             # EigV baseline
|-- configs/               # YAML configuration
|-- data/                  # Log and artifact format docs
|-- docs/                  # Architecture notes
|-- examples/              # Optional log collectors
|-- quick_start/           # Reproducible sample artifacts and scripts
|-- tests/                 # Lightweight unit tests
|-- requirements.txt
|-- pyproject.toml
|-- Makefile
|-- Dockerfile
`-- .env.example
```

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{chen2026every,
  title = {Every Response Counts: Quantifying Uncertainty of LLM-based Multi-Agent Systems through Tensor Decomposition},
  author = {Chen, Tiejin and Yao, Huaiyuan and Chen, Jia and Papalexakis, Evangelos E. and Wei, Hua},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics},
  year = {2026}
}
```

## License

MIT License. See [`LICENSE`](LICENSE) for details.
