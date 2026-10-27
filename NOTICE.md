# Attribution and provenance

The starting research workspace was based on [lyramakesmusic/neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer), local base commit `f79b2196e29aacbad89684938baac8d97b172fd1`. Its neuron browser and intervention work informed these experiments. [Goodfire's neuron research](https://already-clean.goodfire.com/) was an additional inspiration.

The upstream repository did not expose a repository license when checked on 2026-10-27. Accordingly this clean package does not copy its explorer UI, assets, caches, or tracked source code. The browsing, verification, plotting, and reproduction tooling here is newly written. Original experiment scripts are referenced through filenames and SHA-256 fingerprints for audit provenance; they are not relicensed here.

Qwen3.8-27B is provided by Qwen under its own model terms (Apache-2.0 in the official model card). No model weights or tokenizer assets are redistributed. The included model metadata consists of factual configuration fields and fingerprints.

The data consists of the laboratory's recorded intervention artifacts and model-generated excerpts. The legacy source CSVs are retained unchanged and annotated as archival evidence, including fields whose interpretations were rejected. Credits do not imply endorsement by Qwen, Goodfire, Lyra, or OpenAI.

The stopped 2026-10-27 behavioral campaign additionally distributes its newly written frozen instruments and synthetic-prompt generated continuations. These campaign instruments were written for this laboratory study; their inclusion does not expand or relicense the upstream explorer. The complete screen output is provided as a separate release asset, without model weights or tokenizer assets. The compact repository preserves exact source bytes for provenance and generated summaries for browsing.
