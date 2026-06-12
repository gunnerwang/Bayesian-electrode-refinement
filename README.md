# LLM-Guided Bayesian Refinement of Electrode Segmentation Masks

Code release for the paper *"A Bayesian Refinement Method Incorporating Geometric Prior and LLM-Guided Hyperprior for Measurement-Grade Overhang Analysis in X-ray Electrode Inspection"*
by Tianyu Wang et al.

The framework refines coarse electrode instance masks (e.g., from battery
X-ray/CT inspection) with a Bayesian multi-electrode model, optionally guided
by an LLM parameter advisor.

## Repository layout

```
refinement/
├── refiner_llm_guided.py      LLM-guided variant (per-image parameter advisor)
├── refiner_shape_transfer.py  GT shape-feature transfer variant (diagnostic)
├── refiner_assembly.py        GT-guided intelligent-assembly variant (diagnostic)
└── core/                      shared components: Bayesian base refiner,
                               multi-electrode lattice model, prior
                               integration (legacy), pattern refinement,
                               guided refinement (graph-relation modeling
                               lives in the variant entry points)
evaluation/    metric suite (IoU/Dice/Precision/Recall/PixAcc/Boundary-IoU/
               Hausdorff), assembled-mask builder, per-instance evaluation
scripts/       local open-weights LLM serving (Ollama launcher + no-think proxy
               for hybrid reasoning models); calibration builder
docs/          data layout; advisor calibration schema + worked example
```

The LLM parameter advisor lives inside `refiner_llm_guided.py`. It is
**text-only** (numeric pattern statistics in, bounded JSON out — no image
content) and supports two prompts: the **calibrated prompt**, used when a
dataset calibration block is supplied via `--calibration` (recommended; see
[docs/calibration.md](docs/calibration.md)), and the legacy anchored prompt
otherwise.

The components in `core/` are also valid stand-alone refiners.
The GT-guided features of the shape-transfer/assembly variants are diagnostic
regimes and are inactive when no ground truth is provided at inference.

## Installation

```bash
pip install -r requirements.txt
```

## Data layout

See [docs/data_layout.md](docs/data_layout.md). In short:

```
data/
├── Full_Instances/
│   ├── instance_info/<image>_info.json     # per-image instance ids + bboxes
│   ├── repaired_masks/<image>_instance_<id>.png
│   └── masks/<image>_instance_<id>.png      # per-instance GT (evaluation only)
└── Origin/
    ├── images_enhanced/<image>.png
    ├── coarse_masks/<image>.png
    └── masks/<image>.png                    # ground truth (evaluation only)
```

## Quick start

### 1. Batch refinement (no LLM)

```bash
python refinement/refiner_assembly.py \
    --data-dir data/Full_Instances --origin-dir data/Origin \
    --output-dir out/assembly \
    --thickness-factor 1.3 --transform-ratio 0.3 --transform-intensity 0.4
```

### 2. LLM-guided refinement

Any instruction-following language model qualifies (the advisor never sends
images). `--calibration <file>` is optional: it supplies the dataset
calibration block (computed once from a small labeled tuning set using the
evaluation tools of step 3; schema and recipe in
[docs/calibration.md](docs/calibration.md)) and switches the advisor to the
recommended calibrated prompt — without it the legacy anchored prompt is
used. For a first smoke run, append
`--calibration docs/example_calibration.json`.

The backend is selected purely via standard OpenAI environment variables —
for reproducibility we recommend pinned open weights served locally:

```bash
# serve an open-weights model locally (Ollama; see scripts/serve_local_llm.sh)
bash scripts/serve_local_llm.sh qwen3.5:9b-bf16

export OPENAI_BASE_URL=http://127.0.0.1:11435/v1   # no-think proxy (hybrid models)
export OPENAI_API_KEY=ollama
export OPENAI_MODEL=qwen3.5:9b-bf16

python refinement/refiner_llm_guided.py \
    --data-dir data/Full_Instances --origin-dir data/Origin \
    --output-dir out/llm_guided
```

For non-hybrid models (e.g. `llama3.1:8b-instruct-fp16`) point
`OPENAI_BASE_URL` directly at Ollama (`http://127.0.0.1:11434/v1`).

Hosted OpenAI models work through the same path — leave `OPENAI_BASE_URL`
unset and use a dated snapshot for reproducibility:

```bash
export OPENAI_API_KEY=sk-...
export OPENAI_MODEL=gpt-4o-2024-08-06
```

(Note that hosted endpoints are subject to provider-side version drift and
sampling nondeterminism; the pinned local open-weights route above is the
reference configuration for exact reproduction.)

### 3. Evaluation

(Replace `out/llm_guided` with `out/assembly` below if you ran step 1.)

```bash
# per-instance metrics (refined vs baseline vs per-instance GT)
python evaluation/evaluate_instance_masks.py \
    --refined-dir out/llm_guided --baseline-dir data/Full_Instances/repaired_masks \
    --gt-dir data/Full_Instances/masks --output out/instance_metrics.json

# build assembled full-image masks from per-instance outputs
python evaluation/assemble_refined_masks.py \
    --data-dir data/Full_Instances --refined-dir out/llm_guided \
    --output-dir out/llm_guided_assembled
```

## Reproducibility notes

- Advisor calls request temperature 0.1 (a final compatibility retry omits
  sampling parameters); outputs are requested as JSON and clamped to fixed
  ranges; parse failures and empty responses fall back to a deterministic
  rule-based adaptation, and API errors to fixed defaults. With pinned
  local weights the advisor is a fixed function.
- GT-guided code paths are diagnostic only and **off by default**; enable them
  explicitly with `--gt-full-dir <gt mask dir>`.
- A legacy multimodal prior generator exists in `core/` and is disabled by
  default (`enable_llm_priors=False`); the published LLM mechanism is the
  text-only parameter advisor.
- A `.env` file in the working directory is loaded with override enabled and
  takes precedence over shell environment variables for the advisor backend.

## License

Research use only — see [LICENSE](LICENSE).

## Citation

```bibtex
@article{wang2026bayesian,
  title  = {A Bayesian Refinement Method Incorporating Geometric Prior and LLM-Guided Hyperprior for Measurement-Grade Overhang Analysis in X-ray Electrode Inspection},
  author = {Wang, Tianyu and others},
  year   = {2026},
  note   = {to appear; update upon publication}
}
```
