# Claim boundaries

## Supported by this archive

- ECHO-CAVER v1 is a single-runtime FREB/JUDO extension with no inference-time
  router, threshold, ensemble, or post-hoc answer correction.
- Zero initialization is exactly identical to frozen FREB in the recorded
  mechanics test.
- The adapter contains 2,375,680 trainable parameters and its weights have
  SHA-256 `21623bfafc3d329128e2753a5600a9197088ef527deeb44687ea0eacd89f4bb9`.
- Full MMAD-compatible strict normalized 28-cell Overall is 82.416847% over
  39,670 questions.
- JUDO reports 81.20% and AD-Copilot-Thinking reports 82.29% in their standard
  1-shot paper tables. ECHO is numerically +1.216847 and +0.126847 points above
  those values, respectively, under its archived scoring protocol.
- On the 5,600-row component-disjoint paired panel, ECHO improves the point
  estimate by 0.2698 points but the confidence interval includes zero and the
  exact McNemar test is not significant.
- In the later nine-arm publication replication, FREB improves over JUDO by
  0.4593 points and replicated ECHO-full improves over JUDO by 0.5734 points;
  both component-bootstrap intervals exclude zero. ECHO-full improves only
  0.1141 points over FREB and that interval includes zero.
- CE-only is the strongest publication-ablation arm at 81.5821%, but it has no
  Full-MMAD evaluation and is not the released ECHO-v1 checkpoint.

## Not supported

- A statistically confirmed improvement over FREB-CAVER on independent data.
- State-of-the-art status under a harmonized external protocol.
- A claim that Full-MMAD 82.416847% estimates out-of-distribution or production
  generalization.
- A claim that ECHO materially solves anomaly detection; the clean-panel AD
  balanced accuracy changes from 62.9112% to 62.8450%.
- A paired significance or harmonized ranking claim against JUDO or AD-Copilot.
  Their paper-reported row-level predictions were not re-scored with this
  archive's strict parser.
- A claim that the writer-subspace constraint uniquely causes the gain. The
  CE-only and random-subspace controls do not support that exclusivity claim.

## Required reporting language

When quoting the full result, call it a **Full-MMAD-compatible census score**
and report the clean-panel result nearby. Differences from JUDO (`+1.216847`)
and AD-Copilot-Thinking (`+0.126847`) must be called **descriptive cross-paper
deltas**, not statistically verified paired improvements.
