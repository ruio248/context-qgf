# Task3 Context-Q diagnostics

This report contains the four diagnostics requested after the first Task3
Context-Q evaluation report. All diagnostics use the current native QGF
checkpoints and the from-scratch Context-Q checkpoints in
`exp/task3_context_q_from_scratch`.

## 1. Real-checkpoint sampling equivalence

The Context-Q agent was rebuilt with:

```text
native actor parameters
native target-critic parameters
native value parameters
zero ContextInput parameters
context_never_ready (min_context_transitions = 1e9)
```

It was then compared against the native agent using the same observations,
actions, action-noise keys, alpha, and environment reset seed.

| seed | Q max abs diff | gradient max abs diff | alpha=0 action diff | alpha=.04 action diff | rollout return diff | rollout success diff |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 3 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

Result: the zero-context hybrid is exactly equivalent to native QGF with the
same parameters. There is no evidence of a loading, evaluator, or zero-context
fallback mismatch.

## 2. Shared native prefix, then switch critic

Every episode runs its first 20 environment steps with native QGF. The
environment is then branched:

```text
branch N: continue with native QGF
branch C: continue with native actor + Context-Q critic and encoder
```

Both branches start from the same physics state, wrapper bookkeeping, history,
and per-chunk action-noise keys.

| seed | Native success | Context-Q critic success | Delta |
|---:|---:|---:|---:|
| 1 | 70.00% | 40.00% | -30.00 pp |
| 2 | 56.67% | 13.33% | -43.33 pp |
| 3 | 43.33% | 23.33% | -20.00 pp |

Average:

```text
Native after shared prefix: 56.67%
Context critic after shared prefix: 25.56%
Delta: -31.11 pp
```

Result: the degradation remains almost unchanged after removing the first 20
steps as a possible source. The startup stage is therefore not the main cause.

## 3. Latent inference ablation

Actor is fixed to the native actor. The Context-Q critic and encoder use:

```text
mean:   deterministic posterior mean
sample: one posterior sample per action chunk, fixed across denoising
```

| seed | Native success | Context mean success | Context sample success |
|---:|---:|---:|---:|
| 1 | 70.00% | 33.33% | 36.67% |
| 2 | 56.67% | 20.00% | 10.00% |
| 3 | 43.33% | 10.00% | 30.00% |

Average:

```text
Native actor + native Q:       56.67%
Native actor + context Q mean: 21.11%
Native actor + context Q sample: 25.56%
```

Result: posterior sampling changes individual seeds and improves seed 3, but
does not remove the large average gap. Latent inference mode is not the main
explanation.

## 4. Candidate-action ordering

At paired query transitions 20, 40, and 60, three candidate action chunks were
sampled per query. For every pair of candidates, we compared:

```text
predicted ordering: Q(a_i) > Q(a_j)
actual ordering:    MC_return(a_i) > MC_return(a_j)
```

Both critics used the same native continuation for the MC return. Reported
values are pairwise ordering accuracy.

| seed | Native Q accuracy | Context Q accuracy |
|---:|---:|---:|
| 1 | 48.89% | 55.56% |
| 2 | 51.11% | 46.67% |
| 3 | 51.11% | 55.56% |

Average:

```text
Native Q:  50.37%
Context Q: 52.59%
```

Result: both critics are close to chance at ordering these candidate chunks.
The test does not show a clear Context-Q advantage or disadvantage. It is also
too noisy to support a strong causal conclusion by itself, but it does not
explain the closed-loop gap either.

## Overall interpretation

The four diagnostics rule out several simple explanations:

```text
parameter loading mismatch: ruled out
zero-context sampler mismatch: ruled out
first-20-step startup stage: ruled out as the main cause
posterior mean versus sampling: not the main cause
```

The closed-loop degradation remains large and negative under the fixed-actor
and shared-prefix settings. The current evidence still points to the
context-conditioned Q guidance path rather than the actor or evaluator
plumbing.

The remaining candidates are:

1. Training target semantics shared with the original QGF implementation.
2. A mismatch between scalar Q calibration and the action gradient used by
   QGF.
3. Context encoder or conditioning producing an action-dependent signal that
   is harmful even though scalar Q error is lower.

The next experiment should therefore retrain both native QGF and Context-Q with
the same corrected target contract, then repeat the equivalence, shared-prefix,
and closed-loop evaluations. Until then, the current checkpoints should be
treated as diagnostic, not final clean evidence.

## Artifacts

```text
exp/task3_diagnostics_v1/equivalence/seed*/result.json
exp/task3_diagnostics_v1/common_prefix/seed*/result.json
exp/task3_diagnostics_v1/latent/seed*/latent_mean/result.json
exp/task3_diagnostics_v1/latent/seed*/latent_sample/result.json
exp/task3_diagnostics_v1/action_ordering/seed*/result.json
```

## Follow-up diagnostics (v2)

The initial action-ordering run reused the candidate index as the MC rollout
index, so its repeated MC returns were not independent. This was fixed before
the following runs.

### Action ordering, corrected MC seeds

| seed | Native Q accuracy | Context Q accuracy |
|---:|---:|---:|
| 1 | 46.67% | 53.33% |
| 2 | 57.78% | 57.78% |
| 3 | 42.22% | 55.56% |

Average:

```text
Native Q:  48.89%
Context Q: 55.56%
```

Context Q has a small ordering advantage on average, but the sample is small
and the per-seed spread is large. This is suggestive, not conclusive.

### Paired guidance intervention: native chunk versus Context-Q chunk

At each query point the initial action-noise key is shared. Native Q guidance
produces `a_N`; Context-Q guidance produces `a_C`. Both chunks are then
continued with the same native continuation policy and the same continuation
noise sequence.

| seed | Return delta `G(a_C)-G(a_N)` | 95% CI | Success delta |
|---:|---:|---|---:|
| 1 | +26.25 | [-70.47, +124.56] | +3.75 pp |
| 2 | -20.72 | [-115.77, +73.20] | -1.88 pp |
| 3 | -44.69 | [-139.94, +28.62] | -9.38 pp |

Average:

```text
Return delta:  -13.06
Success delta: -2.50 pp
```

At the single-decision level, Context-Q guidance produces only a small average
degradation. The large closed-loop gap therefore appears to accumulate across
many decisions rather than being explained by a consistently terrible action
at the audited early query points.

### Context-off ablation on the trained Context-Q backbone

The Context-Q checkpoint parameters are kept, but `context_ready` is forced to
zero for the whole rollout. The actor is fixed to the native actor.

| seed | Native success | Context backbone with context off | Delta |
|---:|---:|---:|---:|
| 1 | 70.00% | 3.33% | -66.67 pp |
| 2 | 56.67% | 0.00% | -56.67 pp |
| 3 | 43.33% | 0.00% | -43.33 pp |

Average:

```text
Native:                          56.67%
Context backbone with context off: 1.11%
Delta:                           -55.56 pp
```

This is worse than the normal Context-Q result (`24.44%`). The context-trained
critic backbone is therefore not equivalent to the native critic when context
is disabled. This does not by itself prove the backbone is "bad": the backbone
was trained with context active, so disabling context at inference is a
distribution shift. It does show that the problem is not solely the current
context input path.

### Updated interpretation

The follow-up results eliminate or weaken several remaining hypotheses:

```text
evaluator / parameter loading: ruled out
zero-context fallback implementation: ruled out
first-20-step startup: not the main cause
posterior mean versus sampling: not the main cause
single-step guidance intervention: only a small average effect
```

The main remaining hypothesis is a training-level issue:

1. the Context-Q critic backbone is trained under a different value target or
   context-conditional objective than native Q,
2. the context-conditioned backbone does not reduce to a good native-equivalent
   critic when context is unavailable,
3. the degradation accumulates over the closed-loop horizon.

The next decisive experiment is therefore a matched retraining comparison:

```text
keep original target semantics
align native and Context-Q training random streams, data shards, batch indices,
update count, and actor initialization
train one matched seed for localization
```

After that, target-contract correction should be run as a separate factor, with
native and Context-Q both trained under the corrected target.

### v2 artifact locations

```text
exp/task3_diagnostics_v2/action_ordering/seed*/result.json
exp/task3_diagnostics_v2/guidance_intervention/seed*/result.json
exp/task3_diagnostics_v2/context_off/seed*/result.json
```

## MC at alpha=0: pure actor rollout

The two-by-two MC matrix was rerun with:

```text
alpha = 0.0
query rollout = native actor only, no Q guidance
continuation = same alpha=0 policy for the corresponding target
```

### Native continuation target

| seed | Q_native MAE | Q_context MAE | Delta |
|---:|---:|---:|---:|
| 1 | 572.58 | 743.87 | +171.29 |
| 2 | 564.84 | 751.17 | +186.33 |
| 3 | 547.53 | 765.14 | +217.61 |

Average:

```text
Q_native MAE:  561.65
Q_context MAE: 753.39
Delta:         +191.74  (positive favors native Q)
```

### Context continuation target

| seed | Q_native MAE | Q_context MAE | Delta |
|---:|---:|---:|---:|
| 1 | 571.37 | 742.67 | +171.29 |
| 2 | 563.60 | 749.92 | +186.33 |
| 3 | 548.48 | 766.09 | +217.61 |

Average:

```text
Q_native MAE:  561.15
Q_context MAE: 752.89
Delta:         +191.74  (positive favors native Q)
```

### Interpretation

At `alpha=0`, the query distribution is a pure actor/BC rollout rather than a
native Q-guided rollout. Under that distribution, Context-Q is substantially
**worse** than native Q under both continuation targets, and all per-seed
deltas are positive with CIs above zero.

This is a major qualification of the earlier MC result:

```text
alpha=0.04, native Q-guided queries: Context-Q had lower MAE
alpha=0.00, pure actor queries:      Context-Q has much higher MAE
```

The earlier MC advantage is therefore strongly distribution-dependent. It does
not establish that Context-Q is a globally better value function.

### Artifacts

```text
exp/task3_mc_matrix_alpha000/matrix_seed*/result.json
```

## MC calibration across guidance weights

The original paired MC calibration was rerun at the full guidance-weight grid.
For each alpha, native Q and Context-Q were evaluated at the same native-rollout
query points and the same native continuation policy.

Mean MAE across the three seeds:

| alpha | Native Q MAE | Context-Q MAE | Delta | Which is better |
|---:|---:|---:|---:|---|
| 0.000 | 561.65 | 753.39 | +191.74 | Native |
| 0.004 | 122.03 | 308.38 | +186.35 | Native |
| 0.008 | 120.22 | 129.31 | +9.09 | Tie |
| 0.010 | 173.36 | 106.31 | -67.05 | Context-Q |
| 0.020 | 328.56 | 154.50 | -174.07 | Context-Q |
| 0.040 | 277.20 | 180.01 | -97.18 | Context-Q |
| 0.060 | 198.71 | 185.15 | -13.55 | Context-Q (weak) |
| 0.080 | 166.71 | 227.14 | +60.43 | Native |
| 0.100 | 217.53 | 342.81 | +125.28 | Native |
| 0.120 | 314.49 | 445.40 | +130.91 | Native |

Delta is defined as:

```text
Delta = |Q_context - G_MC| - |Q_native - G_MC|
```

Negative values favor Context-Q.

### Interpretation

Context-Q is not globally better calibrated. It is better only in a middle
band of guidance strengths:

```text
alpha <= 0.004: native Q better
alpha ~= 0.008: tie
alpha 0.01-0.06: Context-Q better
alpha >= 0.08: native Q better
```

This strongly supports a distribution-specific explanation: Context-Q's
calibration advantage depends on the trajectory and action distribution
induced by the guidance weight. The earlier `alpha=0.04` MC result was real
for that distribution, but it does not generalize across the full alpha range.

### Artifacts

```text
exp/task3_mc_alpha_sweep_v1/alpha_*/seed*/result.json
```
