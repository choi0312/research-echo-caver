# Reproducibility

## 1. Scope

Offline integrity and score recomputation are self-contained. Autoregressive
reproduction requires separately obtained JUDO, FREB-CAVER, MMAD assets, and
the upstream evaluator because those artifacts are not relicensed here.

## 2. Frozen identities

| Artifact | SHA-256 |
|---|---|
| ECHO adapter weights | `21623bfafc3d329128e2753a5600a9197088ef527deeb44687ea0eacd89f4bb9` |
| ECHO scientific identity | `b538d55b4aaa2d88f0f62f89342741fe1ec807a195db35a2730559c3e5e97bf0` |
| FREB adapter weights | `ac3233dbee2de9dd1aaea4acede275ddc2d7a322427795717d9232291e03b0ed` |
| exact ECHO inference runner | `f160385a1004749485f2dff060e121ac1bf280770d92914171d7613289b8d6a9` |
| full manifest | `61a266302585230610ad58876e76e073ed04f7dccb31c9c61daeaf44a81d4693` |
| full analysis | `68195e2e0fe5e49108b0a48b8b602a1b6e1617611bad2c0b11330ebb6eb8e547` |
| normalized inference contract | `032c460ef670337509502b9cc22f51b85e699da540850ad098313298ec36abd9` |

## 3. Environment

The frozen H200 runtime used:

```text
Python 3.10-compatible environment
torch 2.8.0+cu128
CUDA 12.8
transformers 4.57.3
peft 0.18.1
numpy 2.5.2
attention implementation: SDPA
image size: 512
grid size: 16
batch size per worker: 4
max new tokens: 1024
```

## 4. Obtain upstream dependencies

```bash
git clone https://github.com/woodavid31/JUDO third_party/JUDO
git clone https://github.com/choi0312/FREB-CAVER third_party/FREB-CAVER

huggingface-cli download woodavid31/JUDO \
  --local-dir checkpoints/JUDO

gh release download v0.1.0 \
  --repo choi0312/FREB-CAVER \
  --pattern 'native_deep_residual*' \
  --dir checkpoints/FREB-CAVER
```

Download MMAD from its official distribution and prepare only the paths needed
by your frozen manifest. Do not infer row identity from filename order.

## 5. Train ECHO

`src/train_judo_echo_caver.py` loads helper functions from the separately
cloned FREB/JUDO evaluator through `PYTHONPATH`. A representative command is:

```bash
export JUDO_REPO="$PWD/third_party/JUDO"
export PYTHONPATH="$PWD/third_party/FREB-CAVER/freb_caver:$PWD/src"

python src/train_judo_echo_caver.py \
  --model checkpoints/JUDO \
  --caver-seed checkpoints/FREB-CAVER \
  --train-manifest /path/to/capri/train.jsonl \
  --validation-manifest /path/to/capri/validation.jsonl \
  --data-root /path/to/mmad-assets \
  --output-dir runs/echo-v1 \
  --non-ad-only \
  --projection-mode evidence \
  --basis-rank 32 \
  --lora-rank 16 \
  --lora-alpha 32 \
  --steps 100 \
  --batch-pairs 2 \
  --learning-rate 2e-5 \
  --equivariance-weight 0.25 \
  --retention-weight 1.0 \
  --repair-weight 0.75 \
  --repair-margin 0.25 \
  --seed 20260824
```

The dataset itself is not included. The original train and validation manifest
hashes are retained in `artifacts/training/echo_training_summary.json`.

## 6. Inference

`src/run_judo_manifest.py` is the exact source used by the recorded ECHO runs.
It must be launched with both adapters:

```bash
export JUDO_REPO="$PWD/third_party/JUDO"
export PYTHONPATH="$PWD/third_party/FREB-CAVER/freb_caver:$PWD/src"

python src/run_judo_manifest.py \
  --manifest /path/to/frozen-shard.jsonl \
  --model checkpoints/JUDO \
  --data-root /path/to/mmad-assets \
  --native-adapter-kind caver \
  --native-adapter-directory checkpoints/FREB-CAVER \
  --hybrid-adapter-contract echo-caver-v1 \
  --adapter checkpoints/echo-caver-v1 \
  --batch-size 4 \
  --max-new-tokens 1024 \
  --image-size 512 \
  --grid-size 16 \
  --coverage-threshold 0.02 \
  --attn-implementation sdpa \
  --official-prompt-typo \
  --output runs/shard.jsonl
```

The two H200 recipes in `recipes/` preserve the exact run contract and input
hash checks. Adapt filesystem paths without changing the hashed source,
checkpoint, manifest, or generation parameters.

## 7. Score and verify

For a complete eight-shard census, pass every shard manifest and result pair to
`src/analyze_sharded_adapter_eval.py`. The analyzer fails closed on duplicate or
missing IDs, model/adapter drift, inference-contract drift, or absent native
runtime attestations.

For the archive itself:

```bash
python3 scripts/verify_archive.py
python3 -m unittest discover -s tests -v
```

The original MMAD rows and free generations are intentionally not redistributed.
Consequently this archive verifies the sealed aggregate, checkpoint, code, and
score identities but does not synthesize a fake row-level rerun.

## 8. Publication ablation

The later nine-arm publication ablation is archived separately from the final
ECHO-v1 checkpoint evaluation. Its portable aggregate is
`results/publication_ablation_5600_summary.json`; the private cold-storage
bundle additionally retains the frozen 5,600-row manifest and nine ordered
result JSONLs.

Before reanalysis, verify that every arm has exactly 5,600 unique sample IDs in
manifest order, manifest SHA-256
`cedf23061dc8d95d255fbadd8975de9a7a45a597946b525d83a423f431ee5b33`,
and inference-contract SHA-256
`889b54c354cc99da1522548bde32241f04e8ac20f2bb0df3993cb8e8fced6f28`.
Use 50,000 component bootstrap replicates and seed `20260829`.

Historical absolute paths inside receipts are provenance labels only. Replace
filesystem paths at launch time without changing content hashes or the
scientific contract.
