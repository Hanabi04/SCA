# Reproducing the experiments

This public edition includes seven SLURP prediction groups. FSC sample-level records are prepared locally and are not part of the default CPU reproduction. See the release scope and FSC preparation commands in the root README.



## What the CPU command recomputes

`python reproduce.py` reads `data/experiments.json`, verifies the SHA-256 of every selected prediction archive, checks its sample count, and evaluates each baseline on those same IDs. Task correctness is recomputed by comparing predicted category pairs with the labels. SCA selections are recomputed from the two signed margins, including agreement, parse fallback, and the tie rule.

| Paper result | Inputs and output |
|---|---|
| Table 1 and Figure 2a | Three bundled SLURP full-split prediction archives; `Table 1` and `Comparisons` sheets |
| Table 2 | Orientation margins and score-aware proposals; `Ablations` sheet |
| Correction and corruption analysis | Paired candidate correctness; `Transitions` sheet |
| 1-best-only evidence ablation | Separate judge margins; `Ablations` sheet |
| Three sampling runs | Three bundled SLURP 500-example archives; `Sampling summary` and `All methods` sheets |
| Phi-4-mini transfer | One bundled SLURP 500-example archive; `All methods` sheet |
| Full command/frame and component metrics | Field scores selected from each original candidate record; `Full outputs` sheet |
| Conditional compute | Disagreement counts; `Cost` sheet |
| Judge latency | Ten saved H200 NVL measurements per dataset; `Archived latency` sheet |

The archive allows the reported full-output scores to be aggregated again. It does not contain utterance transcripts or entity strings, so those field scores are not relabeled by the CPU command. Word error rates require the separately obtained reference transcripts and fresh ASR outputs. The timing sheet summarizes the original measurements, rather than timing the current CPU run.

## Statistical conventions

Accuracies are percentages and differences are percentage points. Intervals are paired utterance-level percentile intervals with 10,000 resamples. The historical experiments use bootstrap seed 20260920; the synthetic experiment uses 20260923. Original sample order is retained, because resampling the same integer indices after sorting the rows changes a finite Monte Carlo interval.

Exact two-sided McNemar probabilities are also exported. Repeated-run summaries use the sample standard deviation across the three runs. They are separate from the full-split single-run table. Blanks mean that a method or field was not evaluated for the corresponding experiment.

The archived sampling-run script retained 1-best on a parse failure, whereas the method's rule selects the valid candidate. Two SLURP utterances are affected, one in seed 1 and one in seed 2. `Sampling summary` reproduces the original experiment: SCA and SA-Cand-SCA both average 38.00%. The `Parse fallback` sheet additionally recomputes the valid-candidate rule, under which both average 38.13%. The main table is unchanged. Fresh inference uses the valid-candidate rule. This difference is encoded as `parse_policy` in the experiment registry rather than by changing any prediction or label.

`data/expected.json` stores rounded manuscript accuracies for assertions only. Results are computed from the sample-level records; they are not read from the expected-value file. The main-table intervals are also checked at manuscript precision when the default resample count is used. A checksum, denominator, missing-candidate, or numerical check failure stops the run with an error.

## Dataset partitions

Validation includes 3,118 FSC and 8,690 SLURP utterances. Retained test includes 3,293 and 12,578 utterances after removing 500 previously probed test examples per dataset. Those retained partitions were inspected during method development. The prospective synthetic evaluation sampled 1,000 utterances from 44,918 eligible SLURP synthetic utterances whose actions occur in the training ontology. SCA and its All-5 comparator were fixed before that evaluation; score-aware proposals were examined subsequently.

The bundled SLURP manifests contain stable sample IDs and audio basenames. The synthetic sample and the 500-example subsets are fixed by the released IDs. No resampling is needed to reproduce the reported evaluation populations.

## Baseline definitions

- **1-best, Top-2, All-5:** use the first one, two, or five unique hypotheses in decoder output order.
- **ASR-Mass:** sort by normalized decoding weight and keep hypotheses until cumulative mass reaches 0.90, breaking ties by original index.
- **Score-aware All-5:** use the same five texts and display their normalized decoding weights in the task prompt. This is the paper's implementation of scored N-best prompting, not a redistributed external model.
- **Decision deduplication:** a frozen sentence encoder and trained linear classifier assign coarse decision groups. Group masses determine a 0.95 coverage set; the highest-weight text represents each selected group. The supplied numeric classifier parameters match the archived estimator.
- **Repeated 1-best:** show five copies of the top transcript.
- **SCA:** compare raw 1-best and All-5 proposals; average the signed A/B margins from both candidate orders; a zero margin retains 1-best.
- **SA-Cand-SCA:** replace the All-5 proposal with score-aware All-5 and retain the same judge.
- **1-best-only judge:** keep the original two proposals but show only the top transcript to the judge, without its score.

The main speech and task models are frozen. The small classifier belongs only to decision deduplication; its weights are included so that reproducing this baseline does not require training it again.

## Fresh model inference

Follow [docs/INFERENCE.md](docs/INFERENCE.md). The release keeps the experimental prompts, schemas, checkpoint revisions, scoring direction, and tie behavior. Generation parameters are explicit: temperature 0.6, top-p 0.95, top-k 20, and a 256-token limit.

Fresh sampled generations can differ from archived candidates when the RNG stream, batch scheduling, CUDA kernels, or library versions change. Use the CPU archive for an exact audit of the reported tables and the fresh pipeline for a new execution of the method. Each fresh output records its model revision, seed, and generation configuration.

The packaged CPU suite and unit tests have been run in a clean Windows environment. Full BF16 model inference has not been rerun as part of this release validation. The source includes that execution path; it requires separately downloaded datasets and models and a suitable GPU. Historical judge-only measurements allocated about 16 GiB; allow additional memory for generation. No quantized model is silently substituted.
