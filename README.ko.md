# ECHO-CAVER

**FREB-CAVER를 위한 evidence-tangent residual reader**

[English](README.md) · [Method](METHOD.md) · [Results](RESULTS.md) · [Reproducibility](REPRODUCIBILITY.md) · [Claim boundaries](CLAIMS.md) · [FREB-CAVER](https://github.com/choi0312/FREB-CAVER)

ECHO-CAVER는 공개 [JUDO](https://github.com/woodavid31/JUDO) 및
[FREB-CAVER](https://github.com/choi0312/FREB-CAVER)를 확장한 parameter-efficient
방법론입니다. FREB가 decoder에 비교 증거를 **쓰는** 모듈이라면, ECHO는 이후
decoder가 그 증거를 더 잘 **읽도록** 학습합니다. JUDO와 FREB는 동결되며 하나의
end-to-end model runtime으로 추론합니다.

## Full-MMAD Overall

| 모델 | Full-MMAD Overall | JUDO 대비 | FREB 대비 |
|---|---:|---:|---:|
| JUDO 논문 보고값 | 81.20% | — | — |
| FREB-CAVER | 81.9063% | +0.7063%p | — |
| **ECHO-CAVER v1** | **82.4168%** | **+1.2168%p** | **+0.5105%p** |

ECHO-CAVER는 39,670행 전체를 완료하고 normalized 28-cell macro 82.4168%를
기록했습니다. JUDO는 논문의 standard 1-shot 보고값이며 FREB와 ECHO는 archived
full census 결과입니다. 동일 row-level prediction을 하나의 parser contract로
재평가한 결과가 아니므로 차이는 기술적 benchmark 비교입니다. 정확한 범위는
[Results](RESULTS.md)와 [Claim boundaries](CLAIMS.md)를 참고하십시오.

## FREB 이후에 남은 병목

FREB는 defect-preserving Query–Reference 증거를 계산하고 signed belief를 특정
decoder state에 기록합니다. 그러나 **증거를 쓰는 것만으로 이후 decoder block이
그 증거를 올바르게 읽는다고 보장할 수는 없습니다.** 일반적인 LoRA는 answer
mapping을 학습하면서 visual carrier와 무관한 language-only 방향도 함께 열 수
있어 FREB가 만든 인과적 연결을 약화시킬 수 있습니다.

ECHO-CAVER는 이를 writer–reader alignment 문제로 정의합니다. Frozen FREB
writer가 실제로 생성하는 evidence subspace 안에서만 downstream reader를
학습하여 시각 증거와 decoder adaptation의 연결을 유지합니다.

## 핵심 방법론

FREB writer는 decoder 18, 20, 22, 24, 26층에서 동작합니다. ECHO는 바로 다음
19, 21, 23, 25, 27층의 `o_proj`와 `down_proj`에 rank-16 residual reader를
배치합니다.

각 frozen writer 행렬 `W_i`에서 rank-32 leading left-singular basis `U_i`를
추출하고 LoRA output matrix `B_i`를 다음과 같이 제한합니다.

```text
B_i <- U_i U_i^T B_i
```

Projection은 gradient와 모든 AdamW update 이후에 각각 적용됩니다. 따라서
optimizer state가 지속적인 off-span component를 만들 수 없습니다. LoRA output
path는 zero initialization되어 초기 ECHO가 frozen FREB와 정확히 동일합니다.

ECHO v1은 answer cross-entropy, semantic option equivariance, frozen-FREB
retention, targeted answer repair를 사용합니다. 추론 시에는 JUDO base, FREB
writer, ECHO reader가 하나의 모델에 로드되며 router, threshold, ensemble,
crop policy 또는 post-hoc correction은 사용하지 않습니다.

## 참신성과 차별점

- **Writer-derived adaptation geometry:** Reader subspace를 임의로 선택하지 않고
  실제 배포된 FREB writer weight에서 직접 도출합니다.
- **Evidence-tangent learning:** 기존 task subspace와 직교시키는 O-LoRA 계열과
  달리, 기존 visual evidence carrier 내부에서 의도적으로 학습합니다.
- **Layer-paired writer–reader placement:** FREB가 쓰는 정확한 layer 바로 다음에
  reader를 배치해 architecture causality를 adaptation contract로 만듭니다.
- **두 단계 constraint:** Gradient projection과 post-optimizer projection을 함께
  사용하여 update가 evidence span 밖으로 이탈하는 것을 억제합니다.
- **Single-checkpoint reasoning:** 외부에서 답을 선택하거나 교정하지 않고 모델
  내부의 evidence interpretation 자체를 변화시킵니다.

## 핵심 objective ablation

가장 연구적으로 유의미한 결과는 **CE-only evidence-tangent reader**입니다.
ECHO의 writer-derived subspace와 layer placement는 그대로 유지하면서 auxiliary
retention, equivariance 및 repair loss만 제거합니다.

| Arm | 5,600행 Overall | JUDO 대비 | 95% component-bootstrap CI | Exact McNemar |
|---|---:|---:|---:|---:|
| JUDO | 80.5223% | — | — | — |
| **ECHO CE-only** | **81.5821%** | **+1.0598%p** | **[+0.5243, +1.5947]%p** | **p=0.000096** |

이 결과는 evidence-tangent reader가 단순한 answer CE만으로도 decoder를 개선할
수 있으며, 복합 auxiliary loss가 필수적이지 않고 해당 설정에서는 오히려
과도한 regularization이 될 수 있음을 보여줍니다. 이는 CE-only checkpoint의
Full-MMAD 결과가 아니라 objective에 대한 mechanistic ablation입니다. 전체 9개
arm, 동일 manifest·inference contract, 50,000회 component bootstrap은
[`results/publication_ablation_5600_summary.json`](results/publication_ablation_5600_summary.json)에
보존합니다.

## 핵심 contribution

1. FREB 이후 병목을 **evidence writing과 evidence reading의 분리**로 정의합니다.
2. Decoder update를 frozen visual-evidence writer의 singular span에 연결하는
   writer-conditioned LoRA constraint를 제안합니다.
3. 학습 전 JUDO/FREB 계산을 정확히 보존하면서 2,375,680개 파라미터
   (`0.02853%`)만 학습합니다.
4. Full-MMAD census와 동일-contract paired diagnostic을 함께 제공하고
   component-level bootstrap으로 분석합니다.

## 저장소 구성

| 경로 | 내용 |
|---|---|
| `src/` | ECHO 학습, 추론, manifest 및 scoring 코드 |
| `checkpoints/` | 9.1MB ECHO v1 PEFT adapter |
| `results/` | Full census, paired panel 및 publication ablation 결과 |
| `artifacts/training/` | 고정된 training contract와 mechanics receipt |
| `recipes/` | 기록된 H200 평가 recipe |
| `tests/` | Checkpoint, coverage, score 및 archive identity 테스트 |
| `paper/` | Research archive 원고 |

## 무결성 검증

```bash
python3 scripts/verify_archive.py
python3 -m unittest discover -s tests -v
```

Offline verifier는 v1 체크포인트, 코드, 결과 및 score identity를 검사합니다.
전체 autoregressive 재현에는 각 이용 조건에 따라 JUDO, FREB-CAVER 및 MMAD를
별도로 준비해야 합니다.

## 평가 범위

82.4168% Full-MMAD 결과는 문서화된 개발 설정에서의 전체 behavior census이며
out-of-distribution 일반화 추정치는 아닙니다. 5,600행 publication ablation은
별도의 동일-contract paired 분석입니다. JUDO와의 cross-paper 차이는 동일한
parser 및 inference contract로 row-level 재평가하기 전까지 기술적 비교입니다.

## 상태와 인용

본 저장소는 공개 연구 artifact이며 production anomaly-detection system 또는
채택된 논문이 아닙니다. 결과 인용 시 JUDO와 MMAD 원 연구,
[CITATION.cff](CITATION.cff) 및 [CLAIMS.md](CLAIMS.md)의 해석 범위를 함께
유지하십시오.
