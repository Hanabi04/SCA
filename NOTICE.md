# Third-party resources

The MIT license covers the original project code. Dataset-derived identifiers, labels, and records retain applicable upstream terms. Pretrained models are obtained separately under their own licenses.

FSC is distributed by Fluent.ai. SLURP is distributed by its authors through the [SLURP repository](https://github.com/pswietojanski/slurp). Follow those distributions' data-use and attribution requirements. Benchmark labels and recording identifiers retain their association with those datasets.

Parakeet-RNNT-0.6B is developed by NVIDIA and Suno and distributed under CC BY 4.0, as stated in its [model card](https://huggingface.co/nvidia/parakeet-rnnt-0.6b). Qwen3-8B, Phi-4-mini-instruct, and all-MiniLM-L6-v2 are downloaded from their original model repositories listed in `docs/DATA.md`; their respective license files apply.

NumPy, PyTorch, Transformers, Outlines, NeMo, Hugging Face Hub, and Sentence Transformers are external dependencies. Their source and licenses remain in their installed distributions; this archive does not vendor their implementations. The scientific task prompts and pretrained-model names are part of the experiment specification and are retained.

## Resource licenses

| Resource | Upstream license or terms | Distribution in this archive |
|---|---|---|
| FSC | [Fluent Speech Commands Public License](https://fluent.ai/wp-content/uploads/2021/04/Fluent_Speech_Commands_Public_License.pdf); noncommercial academic use, with restrictions on redistribution | No FSC sample-level records, IDs, or manifests are bundled |
| SLURP | [Text: CC BY 4.0; audio: CC BY-NC 4.0](https://github.com/pswietojanski/slurp/blob/master/LICENSE.txt) | IDs and normalized task labels; no speech or transcripts |
| Parakeet-RNNT-0.6B | [CC BY 4.0](https://huggingface.co/nvidia/parakeet-rnnt-0.6b) | Downloaded separately |
| Qwen3-8B | [Apache-2.0](https://huggingface.co/Qwen/Qwen3-8B/blob/main/LICENSE) | Downloaded separately |
| Phi-4-mini-instruct | [MIT](https://huggingface.co/microsoft/Phi-4-mini-instruct/blob/main/LICENSE) | Downloaded separately |
| all-MiniLM-L6-v2 | [Apache-2.0](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | Downloaded separately |

The benchmark sources are Lugosch et al., *Speech Model Pre-training for End-to-End Spoken Language Understanding* (Interspeech 2019), and Bastianelli et al., *SLURP: A Spoken Language Understanding Resource Package* (EMNLP 2020). Released records normalize task labels and pair them with model predictions and numerical scores. The transformations are described in `docs/DATA.md`.

FSC records are prepared locally from an authorized dataset copy using `prepare_fsc_manifest.py`. This release distributes no FSC sample-level prediction archives or manifests. SLURP records retain their dataset attribution and CC BY 4.0 terms.
