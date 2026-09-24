# Run SCA from audio

This public edition includes seven SLURP prediction groups. FSC sample-level records are prepared locally and are not part of the default CPU reproduction. See the release scope and FSC preparation commands in the root README.



Use Linux with an NVIDIA GPU for the NeMo and task-model pipeline. The CPU table reproduction also runs on Windows and does not require this setup.

## Environments

Create separate Python 3.11 environments for ASR and task inference. This avoids forcing NeMo and the language model stack to share dependency versions.

```sh
python3.11 -m venv .venv-asr
.venv-asr/bin/python -m pip install -r requirements-asr.txt
python3.11 -m venv .venv-inference
.venv-inference/bin/python -m pip install -r requirements-inference.txt
```

Install the appropriate CUDA build of PyTorch for the host. The archived experiments used PyTorch 2.11.0 with CUDA 12.8, Transformers 5.17.0, Outlines 1.3, and NeMo 3.0.0. The dependency files pin the primary libraries; they are not a complete container image. Save `pip freeze` with a fresh run when comparing environments.

The Qwen3-8B model uses BF16. An 8 GB GPU cannot hold the original configuration. Use a GPU with sufficient available memory; 24 GB or more is a practical starting point, with actual requirements depending on prompts and generation. ASR and task inference run as separate processes, releasing memory between stages.

## Obtain the data

Download FSC and SLURP through their official distributions linked in [DATA.md](DATA.md). Extract audio under a local directory. `--audio-root` may point to that directory or a dataset-specific subdirectory. The loader matches the manifest's audio basenames recursively and rejects missing or ambiguous names.

Model checkpoints are downloaded from their original Hugging Face repositories at the recorded revisions. Configure `HF_HOME` if a shared model cache is desired. No credentials are embedded in the scripts.

## Full pipeline

From the repository root, run:

```sh
.venv-inference/bin/python run_pipeline.py \
  --experiment fsc_validation \
  --manifest local_data/fsc_validation.jsonl.gz \
  --audio-root /path/to/fsc \
  --asr-python .venv-asr/bin/python \
  --output runs/fsc_validation
```

The stages write `nbest.jsonl`, `predictions.jsonl`, metadata, and `tables/reproduction_results.xlsx` with CSVs. Stage logs remain in the run directory. Add `--limit 10` for a small setup check. An existing nonempty output directory is rejected, so a new run cannot overwrite previous results.

Other full-split choices are `slurp_validation`, `fsc_retained_test`, `slurp_retained_test`, and `slurp_synthetic`. The pipeline runs 1-best, All-5, Top-2, ASR-Mass, score-aware All-5, decision deduplication, and repeated 1-best, then SCA and its proposal/evidence variants. Some new comparisons therefore go beyond the evaluated cells of the paper; the saved-prediction reproduction preserves the original availability of those cells.

## Separate stages

```sh
.venv-asr/bin/python -m sca.decode \
  --manifest local_data/fsc_validation.jsonl.gz \
  --audio-root /path/to/fsc --output runs/fsc/nbest.jsonl

.venv-inference/bin/python -m sca.infer \
  --dataset fsc --input runs/fsc/nbest.jsonl \
  --output runs/fsc/predictions.jsonl --h1-only

.venv-inference/bin/python -m sca.evaluate_inference \
  --dataset fsc --input runs/fsc/predictions.jsonl \
  --output runs/fsc/tables
```

`sca.infer --model phi4` selects the pinned Phi-4-mini checkpoint. `--seed` controls proposal sampling. `--offline` loads task models from the local cache. The input format is one JSON object per line with `sample_id`, `raw_hypotheses`, and `raw_scores`; include `gold` as the normalized category pair to evaluate accuracy afterward. Gold labels are not passed into model prompts.

For inference from already decoded audio, start at `sca.infer`. Hypotheses are deduplicated case-insensitively, preserving their first occurrence in decoder order, and weights use softmax of the retained sequence scores. The full-output field scores in the paper's archive use the original benchmark evaluation; the fresh exporter reports coarse decisions and transition counts.
