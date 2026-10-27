# Reproduction

## Data audit (CPU, no model loading)

From the repository root:

```bash
python -m pip install numpy
python tools/verify_release.py
python tools/browse.py --entry L20_N2302
python -m unittest discover -s tests -v
```

## GPU probe

Run model experiments on a GPU host with sufficient memory. The data audit above needs no model weights.

```bash
python -m pip install 'torch>=2.8' 'transformers>=5.8,<6' accelerate numpy
python tools/reproduce_probe.py --model /path/to/Qwen3.8-27B --entry L20_N2302 --output replay.json
```

Use a PyTorch CUDA build compatible with the host driver. The audit host supports CUDA 12.8; a CUDA 13 image failed before model loading. The runner uses local files only and verifies the recovered config, generation config, and tokenizer config fingerprints before model loading. It checks the text-tower dimensions and records runtime versions, hardware, prompt token IDs, target tokenization, full baseline/intervention continuations, all configured EOS IDs, and signed token-level effects. The single-entry runner produces an observation, not an automated archive-match verdict; compare its effect and generated excerpts manually. Metadata fingerprints alone are not a weight-shard integrity check.

Historical example generation used a short 12-token cap. Use `--max-new-tokens 12` to compare like with like; increase it when studying complete answers. Differences in dependency versions, attention kernels, numerical precision, and hardware can alter near-tied greedy decoding.

A whole-dictionary replay requires each row of all 16 probes, rather than only the entry's selected example:

```bash
python tools/replay_dictionary.py --model /path/to/Qwen3.8-27B --output-dir /path/to/replay-output
```

The recorded replay used **PyTorch 2.8.0+cu128**, **Transformers 5.21.0**, **Python 3.11.13**, BF16, and an NVIDIA B200. Install the CUDA 12.8 PyTorch build appropriate to the host driver. The reference PyTorch linear-attention path was used because optional optimized kernels were absent. The historical runtime was not fully pinned, so a differing replay cannot be automatically attributed to hardware, numerical precision, or a particular kernel.

The whole-dictionary replay's status is documented in VERIFICATION.md. Calibration arrays are provided as archival inputs; `mean_abs` is not used because the historical accumulator divided by sequence count rather than token count.
