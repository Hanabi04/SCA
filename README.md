# Selective Contrastive Adjudication

SCA compares the task decisions produced from a single ASR transcript and an N-best list. It returns their common decision when they agree. Otherwise, the same language model scores both candidate orders and averages the signed margins to select a decision.

This repository contains the method, task baselines, experiment manifests, and a compact prediction archive for the accompanying preprint. The models are frozen; SCA requires no fine-tuning.

## Reproduce the reported results

Install Python 3.11 or newer and clone or download this repository to a writable directory. On Windows, double-click **`reproduce.cmd`**. On Linux or macOS, run:

```sh
sh reproduce.sh
```

The launcher creates `.venv-reproduce`, installs the CPU dependency, and recomputes all seven bundled SLURP experiment groups. Internet access is needed for the first dependency installation. Subsequent runs reuse that environment. No API key, speech data, model download, or GPU is needed for this mode.

Open `reproduction_results/<run>/reproduction_results.xlsx`. The workbook contains the main comparison, ablations, transition counts, paired intervals, repeated-run statistics, and full-output scores. Each sheet also has a CSV counterpart. Numeric results are numeric cells, with filters and a frozen header. `metrics.json` retains full numerical precision; `run.json` records the environment and run status.

The main checks are:

| Split | Dataset | Utterances | SCA accuracy |
|---|---|---:|---:|
| Validation | SLURP | 8,690 | 34.29% |
| Retained test | SLURP | 12,578 | 32.57% |
| Synthetic | SLURP | 1,000 | 39.60% |

The `Parse fallback` sheet documents a two-utterance difference between the archived sampling-run implementation and the valid-candidate fallback rule. Both variants are recomputed; see [REPRODUCIBILITY.md](REPRODUCIBILITY.md). This does not change Table 1.

This command evaluates saved predictions and recomputes SCA decisions from their two orientation margins. It does not rerun the speech or language models. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for experiment coverage and [docs/INFERENCE.md](docs/INFERENCE.md) to run the models from audio.

For an existing environment:

```sh
python -m pip install -r requirements.txt
python reproduce.py
python -m unittest discover -s tests -v
```

An individual experiment can be selected with `--experiment slurp_validation`. The default is 10,000 paired bootstrap resamples, matching the paper. `--bootstrap 100` is useful for a quick software check; its confidence intervals are not the paper intervals. Each run creates a new directory, leaving previous results intact.

## Code layout

```text
sca/                 Selection, inference, statistics, and spreadsheet export
configs/             Task schemas and decision-deduplication metadata
data/predictions/    Compressed sample-level decisions and judge margins
data/manifests/      Evaluation IDs and audio basenames
data/models/         Small linear classifiers used by the deduplication baseline
tests/               Decision, statistics, archive, and export tests
docs/                Inference setup and data documentation
reproduce.py         CPU result reproduction
run_pipeline.py      Audio-to-result pipeline
```

All baselines are named explicitly: 1-best, All-5, Top-2, ASR-Mass, score-aware All-5, decision deduplication, and repeated 1-best. SCA, its two single-order variants, the 1-best-only judge, and SA-Cand-SCA appear separately. The pairwise oracle uses ground-truth labels and is reported as an upper bound.

## License and data

The original project code is released under the [MIT license](LICENSE). Dataset-derived identifiers, labels, and evaluation records retain applicable upstream terms; their scope is described in [NOTICE.md](NOTICE.md). Audio datasets, pretrained models, and their licenses are separate; see [NOTICE.md](NOTICE.md). The archive contains no audio or third-party model weights. The small decision-deduplication classifiers are provided as numeric NPZ arrays rather than serialized Python objects.

## Dataset coverage

This edition includes the seven SLURP prediction groups. The one-click CPU command recomputes those results and exports Excel and CSV tables. FSC sample-level prediction archives and manifests are not distributed because their redistribution permission has not been established. The paper's FSC aggregate results remain reported in the manuscript; they are not reconstructed by the default public CPU command.

FSC inference code, task schemas, and the small trained deduplication classifier are included. Obtain the official dataset under its terms, prepare a local manifest, and run fresh inference:

```sh
python prepare_fsc_manifest.py --csv /path/to/fsc/data/valid_data.csv --output local_data/fsc_validation.jsonl.gz
python run_pipeline.py --experiment fsc_validation --manifest local_data/fsc_validation.jsonl.gz --audio-root /path/to/fsc/wavs --output runs/fsc_validation
```

Install the separate inference/ASR dependencies described in `docs/INFERENCE.md` first. Fresh stochastic generation can differ from the archived predictions. For the retained-test protocol, use the original local exclusion list with `--exclude-ids`; running the full official test CSV evaluates a different population. The public release does not distribute that FSC list. No raw speech or utterance transcripts are bundled.

## Citation

Guorui He, Zelin Jin, Jiabin Fan. *Selective Adjudication of ASR N-Best Decisions for Task-Oriented Voice Agents*. 2026.

See [CITATION.cff](CITATION.cff) for machine-readable citation metadata.
