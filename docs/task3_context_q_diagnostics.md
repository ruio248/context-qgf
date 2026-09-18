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
