# Task3 Context-Q evaluation report

> **Status:** This document records the older from-scratch Context-Q study.
> Its MC continuation and C-norm-related diagnostics are not the primary
> protocol for `finetune/context-q-adapter`, and its numbers must not be used
> as evidence for the new 500k / 530k / 530k three-arm adapter comparison.

## Scope

This report summarizes the current Task3 evaluation results for:

- Native QGF, using the existing native checkpoints.
- Context-Q, using the from-scratch clean-branch checkpoints.
- Paired MC value calibration.
- Paired closed-loop policy success.
- Fixed-actor Q swap and alpha-sweep ablations.

The results below describe the current checkpoints and evaluation protocol. They
are not yet a final "clean native QGF from scratch" comparison because the
native checkpoints were produced by the earlier `qgf-native` runs, including
resume/recovery history.

## Checkpoints and run locations

Context-Q checkpoints:

```text
exp/task3_context_q_from_scratch/context-qgf-clean/task3_context_q_seed01/...
exp/task3_context_q_from_scratch/context-qgf-clean/task3_context_q_seed02/...
exp/task3_context_q_from_scratch/context-qgf-clean/task3_context_q_seed03/...
```

Each run contains `params_100000.pkl` through `params_500000.pkl`.

Native checkpoints:

```text
/home/lrh/qgf-native/artifacts/runs/native_qgf_cube_triple_task3_3seed/seed01/...
/home/lrh/qgf-native/artifacts/runs/native_qgf_cube_triple_task3_3seed/seed02/...
/home/lrh/qgf-native/artifacts/runs/native_qgf_cube_triple_task3_3seed/seed03/...
```

## MC calibration protocol

Original paired MC calibration:

```text
environment: cube-triple-play-singletask-task3-v0
alpha: 0.04
episodes per seed: 10
query transitions: 20, 40, 60
MC rollouts per query: 8
continuation target: frozen native QGF
```

Two-by-two MC matrix:

```text
continuation targets:
  G_native  = frozen native QGF continuation return
  G_context = Context-Q continuation return

critics:
  Q_native
  Q_context
```

The matrix compares all four combinations on the same query points.

## MC calibration result: native continuation

Per-seed results:

| seed | Q_native MAE | Q_context MAE | Delta | 95% CI |
|---:|---:|---:|---:|---|
| 1 | 320.03 | 239.24 | -80.79 | [-140.18, -18.68] |
| 2 | 237.18 | 139.07 | -98.11 | [-160.29, -35.70] |
| 3 | 274.37 | 161.72 | -112.65 | [-172.31, -55.67] |

Aggregate:

| target | Q_native MAE | Q_context MAE | Delta |
|---|---:|---:|---:|
| Native QGF continuation | 277.20 | 180.01 | -97.18 |

Negative Delta favors Context-Q. All three seeds favor Context-Q.

## MC calibration result: Context-Q continuation

Per-seed results:

| seed | Q_native MAE | Q_context MAE | Delta | 95% CI |
|---:|---:|---:|---:|---|
| 1 | 280.04 | 210.25 | -69.79 | [-135.78, +0.96] |
| 2 | 266.22 | 134.79 | -131.43 | [-164.71, -96.04] |
| 3 | 252.53 | 144.09 | -108.43 | [-178.74, -31.42] |

Aggregate:

| target | Q_native MAE | Q_context MAE | Delta |
|---|---:|---:|---:|
| Context-Q continuation | 266.26 | 163.04 | -103.22 |

Context-Q is still better calibrated when each critic is evaluated against its
own policy's continuation return. The MC target mismatch therefore does not
explain the closed-loop behavior.

## Closed-loop protocol

Paired closed-loop evaluation:

```text
environment: cube-triple-play-singletask-task3-v0
alpha: 0.04
episodes per seed: 30
shared episode reset seeds
shared per-chunk action-noise keys
paired action-space reset seed
```

## Full closed-loop comparison

Full comparison: Context-Q actor + Context-Q critic versus native actor +
native critic.

| seed | Native success | Context-Q success | Delta |
|---:|---:|---:|---:|
| 1 | 70.00% | 46.67% | -23.33 pp |
| 2 | 56.67% | 13.33% | -43.33 pp |
| 3 | 43.33% | 13.33% | -30.00 pp |

Aggregate:

| method | Success | Return |
|---|---:|---:|
| Native QGF | 56.67% | -1434.37 |
| Context-Q | 24.44% | -1452.91 |
| Delta | -32.22 pp | -18.54 |

The closed-loop success difference is large and negative despite the better MC
calibration.

## Fixed-actor Q swap

This ablation keeps the native actor fixed and swaps only the critic:

```text
native actor + native Q
versus
native actor + context Q
```

| seed | Native actor + native Q | Native actor + context Q | Delta |
|---:|---:|---:|---:|
| 1 | 70.00% | 33.33% | -36.67 pp |
| 2 | 56.67% | 20.00% | -36.67 pp |
| 3 | 43.33% | 10.00% | -33.33 pp |

Aggregate:

| method | Success | Return |
|---|---:|---:|
| Native actor + native Q | 56.67% | -1434.37 |
| Native actor + context Q | 21.11% | -1359.60 |
| Delta | -35.56 pp | +74.77 |

The closed-loop degradation is therefore primarily attributable to the
context-conditioned Q guidance, not to actor differences.

## Alpha sweep

Closed-loop alpha sweep, averaged across three seeds:

| alpha | Native success | Context-Q success | Delta |
|---:|---:|---:|---:|
| 0.000 | 0.00% | 0.00% | 0.00 pp |
| 0.004 | 1.11% | 4.44% | +3.33 pp |
| 0.008 | 20.00% | 11.11% | -8.89 pp |
| 0.010 | 25.56% | 17.78% | -7.78 pp |
| 0.020 | 56.67% | 17.78% | -38.89 pp |
| 0.040 | 56.67% | 24.44% | -32.22 pp |
| 0.060 | 57.78% | 13.33% | -44.44 pp |
| 0.080 | 38.89% | 8.89% | -30.00 pp |
| 0.100 | 20.00% | 1.11% | -18.89 pp |
| 0.120 | 14.44% | 0.00% | -14.44 pp |

Native QGF's best region is approximately `alpha=0.02-0.06`. Context-Q's best
alpha is `0.04`, with mean success `24.44%`, while native reaches `57.78%` at
`alpha=0.06`. Alpha tuning therefore does not close the gap.

## Bias analysis of the MC improvement

For each query, define:

```text
ideal correction   = G_MC - Q_native
context correction = Q_context - Q_native
```

The context correction has a much smaller variance than the ideal correction:

| seed | ideal correction std | context correction std | correlation |
|---:|---:|---:|---:|
| 1 | 274.86 | 28.80 | 0.36 |
| 2 | 189.07 | 52.44 | 0.49 |
| 3 | 213.72 | 46.80 | 0.48 |

The MC improvement is therefore mostly a systematic bias correction, rather
than a state/action-conditioned improvement in the critic.

## Interpretation

The following explanations are currently ruled out:

- Alpha is not the cause: the gap persists across the full sweep.
- Continuation-target mismatch is not the cause: Context-Q remains better
  calibrated under both native and Context-Q continuation targets.
- Seed pairing is not the cause: the closed-loop evaluator now shares episode
  reset seeds, per-chunk action-noise keys, and the action-space reset seed.

The current evidence points to a mismatch between:

```text
better scalar Q calibration
and
worse action-gradient quality for QGF guidance.
```

Q values can become more accurate largely by shifting a bias, while the
guidance direction `grad_a Q(s,a)` can remain unchanged or become worse.

## Recommended next diagnostics

1. Compare `||grad_a Q_context||` and `||grad_a Q_native||` at the same
   denoiser query points.
2. Measure cosine similarity between the two Q gradients.
3. Measure action saturation under the one-Euler-step QGF approximation.
4. Compare Q error and Q-gradient error on states visited by the Context-Q
   closed-loop policy.

These diagnostics should determine whether Context-Q is learning a useful
value estimate whose action gradient is nevertheless unsuitable for QGF.

## Result locations

Original MC:

```text
exp/task3_mc_clean/mc_seed*/result.json
exp/task3_mc_clean/aggregate/result.json
```

Two-by-two MC matrix:

```text
exp/task3_mc_matrix_alpha004/matrix_seed*/result.json
```

Closed-loop alpha=0.04:

```text
exp/task3_closed_loop_alpha004_v3/closed_loop_seed*/result.json
```

Fixed-actor Q swap:

```text
exp/task3_qswap_alpha004/qswap_seed*/result.json
```

Alpha sweep:

```text
exp/task3_closed_loop_sweep_alpha_grid_v1/seed*/summary.csv
exp/task3_closed_loop_sweep_alpha_grid_v1/seed*/result.json
```

## Relevant commits

```text
5d12208 Add two-by-two MC calibration matrix
5c8110c Add paired closed-loop alpha sweep
2f6ed92 Support fixed-actor Q ablation in closed-loop evaluator
8504420 Seed paired action space in closed-loop evaluator
6a4782d Use per-chunk keys and drop overstrict pre-context check
968008d Use per-chunk paired keys in closed-loop evaluator
```
