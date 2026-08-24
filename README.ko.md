# [Research] ECHO-CAVER: FREB-CAVER의 시각 증거 접공간 잔차 리더

ECHO-CAVER는 JUDO와 FREB-CAVER 본체를 동결한 채, FREB가 작성한 시각 비교
증거를 다음 LLM block에서 읽도록 만드는 rank-16 LoRA 확장입니다. LoRA-B의
출력 방향은 각 FREB writer의 상위 rank-32 left-singular span 안으로 제한됩니다.
외부 router, threshold, ensemble, crop 또는 사후 정답 보정 없이 하나의 결합
checkpoint로 전체 MMAD task를 추론합니다.

## 핵심 결과

| 모델 | AD | DC | DL | DD | DA | OC | OA | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| JUDO (논문) | 65.04 | 74.74 | 73.01 | 84.56 | 89.41 | 94.04 | 87.58 | 81.20 |
| AD-Copilot (논문) | 73.64 | 67.89 | 64.08 | 80.60 | 85.91 | 91.06 | 87.78 | 78.71 |
| AD-Copilot-Thinking (논문) | **73.95** | 74.29 | **76.40** | 84.92 | 86.93 | 91.86 | 87.67 | 82.29 |
| **ECHO-CAVER v1** | 65.8231 | **76.7698** | 75.3531 | **86.1556** | **89.6094** | **95.1548** | **88.0521** | **82.4168** |

ECHO의 Overall은 JUDO 논문보다 **+1.2168%p**, AD-Copilot-Thinking
논문보다 **+0.1268%p** 높습니다. 다만 외부 두 모델의 값은 각 논문의 standard
1-shot 보고치이고 ECHO는 이 archive의 strict 39,670-row/28-cell 정규화
점수이므로, 이는 paired significance가 아닌 기술적 cross-paper 비교입니다.

Clean panel의 component-cluster 95% CI는 `[-0.0937, +0.6317]`%p이고
McNemar `p=0.13321`이므로 통계적으로 확정적인 향상은 아닙니다. Full 결과는
benchmark-targeting 및 학습 중복 component가 포함된 성능 감사이며, 독립적인
일반화 수치로 주장하지 않습니다.

## 방법 요약

- FREB writer layer: `18, 20, 22, 24, 26`
- ECHO reader layer: `19, 21, 23, 25, 27`
- 대상 module: `o_proj`, `down_proj`
- LoRA rank/alpha: `16/32`
- evidence basis rank: `32`
- 학습: 비AD 963 pair, validation 184 pair, 100 step, AdamW `2e-5`
- 목적함수: 정답 CE + option permutation equivariance + frozen-FREB KL +
  wrong-answer repair margin
- 학습 parameter: 2,375,680개, 전체 8.328B의 0.0285%

세부 방법은 [METHOD.md](METHOD.md), 실험은 [RESULTS.md](RESULTS.md), 재현은
[REPRODUCIBILITY.md](REPRODUCIBILITY.md), 허용 가능한 주장 범위는
[CLAIMS.md](CLAIMS.md)를 참고하십시오.
