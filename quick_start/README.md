# Quick Start

This folder contains ready-to-run MATU sample artifacts. You can get the main
results immediately, or optionally regenerate intermediate files step by step.

The quick start includes two settings:

| Setting | Dataset | Agent Framework / Log Source | LLM | Included Artifacts |
| --- | --- | --- | --- | --- |
| MATH + Qwen2.5 | MATH | Included Qwen2.5 conversation logs with role-wise trajectories. | Qwen2.5-7B-Instruct | Conversation logs, MATH labels, Qwen3 role embeddings, MATU fit curves, scalar uncertainty, and SAUP-Multiple baseline scores. |
| MMLU + AutoGen + Qwen2.5 | MMLU | Included AutoGen multi-agent logs with analyst, verifier, and star roles. | Qwen2.5-7B-Instruct | Conversation logs, MMLU labels, Qwen3 role embeddings, MATU fit curves, and SAUP-Multiple baseline scores. |

The important point: **the logs, labels, fit dictionaries, scalar uncertainty,
baseline scores, and zipped reference embeddings are already provided.** You do
not need to re-embed logs or rerun CP-2 unless you want to inspect the pipeline
internals.

## Environment

Quick-start scripts read local settings from:

```text
quick_start/.env
```

Create it from the template:

```bash
cp quick_start/.env.example quick_start/.env
```

On Windows PowerShell:

```powershell
Copy-Item quick_start/.env.example quick_start/.env
```

Important variables:

```text
OPENAI_API_KEY=                 # only needed for CAMEL/GPT log generation
MATU_OPENAI_MODEL=GPT_4O
MATU_QWEN_MODEL=Qwen/Qwen2.5-7B-Instruct
MATU_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
MATU_MODEL_CACHE=               # optional local HF cache/model path
MATU_MAX_RANK=50
```

The included reference evaluation does not require `OPENAI_API_KEY`, a GPU, or
any model downloads. `OPENAI_API_KEY` is only needed for optional OpenAI/CAMEL
log generation; `MATU_MODEL_CACHE` is useful if you re-embed logs from a local
Hugging Face cache or model path; `MATU_MAX_RANK` controls the optional CP-2
wrappers.

## Included Files

| File | Description |
| --- | --- |
| `data/conversation_logs_Math_qwen2.5.json` | Ready-to-use MATH repeated conversation logs. |
| `data/conversation_logs_MMLU_Autogen_qwen2.5.json` | MMLU AutoGen conversation log sample for the quick-start setting. |
| `data/embeddings_Math_qwen2.5_qwen3.zip` | Zipped Qwen3 role embedding matrices for MATH. |
| `data/embeddings_MMLU_Autogen_qwen2.5.zip` | Zipped AutoGen analyst, verifier, and star embedding matrices for MMLU. |
| `data/embedding_metadata.json` | Embedding model metadata. |
| `results/fit_dict_Math_qwen2.5_qwen3embedding.pkl` | Ready-to-use MATU fit curves for MATH. |
| `results/uncertainty_Math_qwen2.5.pkl` | Ready-to-use scalar MATU uncertainty for MATH. |
| `results/accuracy_dict_Math_qwen2.5.pkl` | MATH repeated-run correctness labels. |
| `results/fit_dict_MMLU_Autogen_qwen2.5.pkl` | Ready-to-use MATU fit curves for MMLU AutoGen. |
| `results/accuracy_dict_MMLU_Autogen_qwen2.5.pkl` | MMLU AutoGen repeated-run correctness labels. |
| `results/saup_scores_Math_qwen2.5.pkl` | Ready-to-use MATH SAUP-Multiple baseline scores. |
| `results/saup_scores_MMLU_Autogen_qwen2.5.pkl` | Ready-to-use MMLU AutoGen SAUP-Multiple baseline scores. |

## Quick-Start Workflow

| Step | Command | What It Does | Required? | Output |
| --- | --- | --- | --- | --- |
| 00: Generate logs | `python quick_start/code/00_generate_logs_*.py` | Collects new logs when users want to replace the included samples. | Optional; included logs are already in `data/`. | New logs under `quick_start/generated/`. |
| 01: Embed logs | `python quick_start/code/01_embed_reference_logs.py` | Stages the provided MATH embeddings by default, or recomputes Qwen3 embeddings when external model settings are enabled. | Optional; reference embeddings are already zipped in `data/`. | `quick_start/generated/embeddings/*.pkl`. |
| 02: Run CP-2 | `python quick_start/code/02_run_cp2_from_generated_embeddings.py --embedding-source provided` | Recomputes MATU fit curves from provided, generated, or explicitly passed embeddings. | Optional; reference fit curves are already in `results/`. | `quick_start/generated/results/matu_scores.pkl` and `fit_dict_generated.pkl`. |
| 03: Convert fit curves | `python quick_start/code/03_fit_to_uncertainty_*.py` | Converts MATU fit curves to scalar uncertainty. | Optional for included files; needed after generated CP-2. | Included or generated uncertainty `.pkl`. |
| 04: Evaluate MATU | `python quick_start/code/04_evaluate_reference_results.py --sample all` | Loads included MATU results and labels, then prints MATH and MMLU metrics. | Yes, for fastest verification. | Console AUROC/AUARC. |
| 05: Evaluate baselines | `python quick_start/code/05_evaluate_baselines.py --sample all` | Loads included SAUP-Multiple results for MATH and MMLU. | Optional comparison. | Console AUROC/AUARC. |

## 00: Optional Generate New Logs

The included logs are already available in `quick_start/data/`, so generation is
not needed for the quick-start result. Run these only to collect new logs.

**Important:** `00_generate_logs_camel_gpt.py` requires `OPENAI_API_KEY` in
`quick_start/.env` and the example dependencies from `pip install -e ".[examples]"`.
`00_generate_logs_hf_qwen.py` requires local or downloadable
Hugging Face model access through `MATU_QWEN_MODEL` and, if needed,
`MATU_MODEL_CACHE`.

```bash
# 00: MATH + CAMEL/OpenAI example -> new logs under quick_start/generated/
python quick_start/code/00_generate_logs_camel_gpt.py

# 00: MATH + HF/Qwen example -> new logs under quick_start/generated/
python quick_start/code/00_generate_logs_hf_qwen.py
```

The CAMEL/OpenAI script needs `OPENAI_API_KEY`; the HF/Qwen script needs local
model access.

## 01: Optional Re-Embed Logs

Use this path when you want a script to prepare Step `01` outputs. By default,
it stages the provided MATH embeddings into `quick_start/generated/embeddings/`
without downloading a model. Set `MATU_RUN_EXTERNAL=1` or `MATU_MODEL_CACHE` if
you want to recompute embeddings with `Qwen/Qwen3-Embedding-0.6B`; a GPU is
faster for that path.

**Important:** Recomputing Step `01` loads `Qwen/Qwen3-Embedding-0.6B`. Make
sure the core dependencies are installed, and set `MATU_MODEL_CACHE` in
`quick_start/.env` if you want to use a local Hugging Face cache/model path.

```bash
# 01: provided MATH + Qwen3 embeddings -> staged Step 01 outputs
python quick_start/code/01_embed_reference_logs.py
```

Output:

```text
quick_start/generated/embeddings/user_embedding_matrices.pkl
quick_start/generated/embeddings/assistant_embedding_matrices.pkl
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

## 02: Optional Rerun CP-2

Use this path when you want to verify MATU/CP-2 from raw embeddings but do not
want to download the embedding model. The public `fit_dict` is already provided,
so this is not required for the fastest result check.

```bash
# 02: provided MATH + Qwen3 embeddings -> regenerated MATU fit curves
python quick_start/code/02_run_cp2_from_generated_embeddings.py \
  --embedding-source provided \
  --max_rank 50
```

If you recomputed or staged Step `01` into `quick_start/generated/embeddings/`,
use the generated source. To pass your own embedding files, use `--embeddings`
with one pickle per role.

```bash
# 02: MATH + generated Qwen3 embeddings -> regenerated MATU fit curves
python quick_start/code/02_run_cp2_from_generated_embeddings.py \
  --embedding-source generated

# 02: custom role embeddings -> regenerated MATU fit curves
python quick_start/code/02_run_cp2_from_generated_embeddings.py \
  --embeddings path/to/user_embedding_matrices.pkl path/to/assistant_embedding_matrices.pkl
```

## 03: Convert Fit Curves To Uncertainty

The reference script rebuilds the included MATH uncertainty file in
`quick_start/results/`. The generated script converts optional CP-2 outputs
under `quick_start/generated/results/`.

```bash
# 03: included MATH fit_dict -> included scalar uncertainty
python quick_start/code/03_fit_to_uncertainty_reference.py

# 03: regenerated MATH fit_dict -> regenerated scalar uncertainty
python quick_start/code/03_fit_to_uncertainty_generated.py
```

## 04: Evaluate MATU

The reference evaluator is the fastest check: it reads included `.pkl` files and
does not download models, call APIs, regenerate embeddings, or rerun CP-2.

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

## 05: Evaluate Baselines

Evaluate the included SAUP-Multiple reference results:

```bash
# 05: included MATH + Qwen2.5 SAUP-Multiple baseline
python quick_start/code/05_evaluate_baselines.py --sample math-qwen

# 05: included MMLU + AutoGen + Qwen2.5 SAUP-Multiple baseline
python quick_start/code/05_evaluate_baselines.py --sample mmlu-autogen-qwen

# 05: both included SAUP-Multiple baselines
python quick_start/code/05_evaluate_baselines.py --sample all
```

Expected output:

```text
MATH + Qwen2.5 + SAUP-Multiple:
  tasks: 400
  AUROC: 0.6078
  AUARC: 0.8722

MMLU + AutoGen + Qwen2.5 + SAUP-Multiple:
  tasks: 400
  AUROC: 0.7194
  AUARC: 0.8589
```

EigV code is provided in the root baseline script. Its result file is not
included in this quick start:

```bash
python baselines/eigv.py \
  --logs quick_start/data/conversation_logs_Math_qwen2.5.json \
  --mode final \
  --out quick_start/generated/results/eigv_final.pkl
```

## Expected Results

| Setting | Method | Source | AUROC | AUARC | Command |
| --- | --- | --- | --- | --- | --- |
| MATH + Qwen2.5 | MATU | Paper Table 1 | 0.7089 | 0.9064 | Paper reference. |
| MATH + Qwen2.5 | MATU | Included quick-start artifact | 0.7205 | 0.9017 | `python quick_start/code/04_evaluate_reference_results.py --sample math-qwen` |
| MATH + Qwen2.5 | SAUP-Multiple | Included quick-start baseline | 0.6078 | 0.8722 | `python quick_start/code/05_evaluate_baselines.py --sample math-qwen` |
| MMLU + AutoGen + Qwen2.5 | MATU | Paper Table 2 | 0.7315 | 0.8833 | Paper reference. |
| MMLU + AutoGen + Qwen2.5 | MATU | Included quick-start artifact | 0.7315 | 0.8834 | `python quick_start/code/04_evaluate_reference_results.py --sample mmlu-autogen-qwen` |
| MMLU + AutoGen + Qwen2.5 | SAUP-Multiple | Included quick-start baseline | 0.7194 | 0.8589 | `python quick_start/code/05_evaluate_baselines.py --sample mmlu-autogen-qwen` |

The MATH artifact was re-run for public release packaging, so it is close to
but not bit-for-bit identical to the paper table. The MMLU AutoGen artifact is
the original artifact used for the paper result, so it matches Table 2 up to
display rounding.
