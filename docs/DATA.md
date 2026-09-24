# Data and model sources

This public edition includes seven SLURP prediction groups. FSC sample-level records are prepared locally and are not part of the default CPU reproduction. See the release scope and FSC preparation commands in the root README.



## Benchmarks

- [Fluent Speech Commands](https://fluent.ai/fluent-speech-commands-a-dataset-for-spoken-language-understanding-research/): Lugosch et al., *Speech Model Pre-training for End-to-End Spoken Language Understanding*, Interspeech 2019.
- [SLURP](https://github.com/pswietojanski/slurp): Bastianelli et al., *SLURP: A Spoken Language Understanding Resource Package*, EMNLP 2020. Obtain the real and synthetic audio and annotations using that repository's current instructions and license terms.

This release redistributes no audio, utterance transcripts, or entity-value strings. `data/manifests` stores sample identifiers, audio basenames, and task categories. `data/predictions` stores categorical model decisions, numerical field scores, and signed judge margins. These records support result checking without downloading speech.

Prediction rows have the following fields:

| Field | Meaning |
|---|---|
| `sample_id` | Original benchmark recording identifier |
| `gold` | Normalized object/action or scenario/action pair |
| `predictions.<method>.decision` | Predicted pair, or null on parse failure |
| `predictions.<method>.scores` | Original per-example field evaluation |
| `margins.<variant>` | Two signed log-probability margins favoring the alternative |

Margins are required only when both decisions parse and disagree. SHA-256 checksums and the expected number of samples appear in `data/experiments.json`. Gzip archives use UTF-8 JSON Lines. No Python pickle is required to load a released data file.

## Frozen models

| Component | Repository | Revision |
|---|---|---|
| ASR | [nvidia/parakeet-rnnt-0.6b](https://huggingface.co/nvidia/parakeet-rnnt-0.6b) | `1b6b548f70b93d2410c3d13cc0654cab300f06ef` |
| Main task model | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | `b968826d9c46dd6066d109eabc6255188de91218` |
| Second task model | [microsoft/Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct) | `cfbefacb99257ffa30c83adab238a50856ac3083` |
| Deduplication encoder | [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | See each dataset's `configs/*_dedup.json` |

The task schemas enumerate the training-set labels. The small linear classifiers in `data/models` contain coefficient matrices, intercepts, and class indices from the original estimators. Their encoder and label mapping are recorded alongside them. Pretrained model weights remain with their original distributors.
