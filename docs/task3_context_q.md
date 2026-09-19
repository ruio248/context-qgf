# Task3 clean Context-Q experiment

This branch contains both the original from-scratch architecture comparison and
a checkpoint-preserving Context-Q adapter finetuning protocol.  They answer
different questions and must not be merged into one result table.

The two training arms use the same Task3 OGBench source, seed, 500,000 offline
updates, batch size, QGF actor configuration, guidance alphas, and checkpoint
schedule:

```bash
export OGBENCH_DATA_DIR=/data/ogbench
./scripts/train_task3_native_qgf.sh 1
./scripts/train_task3_context_q.sh 1
```

Run three matched seeds by changing the final argument to `1`, `2`, and `3`.
The Context-Q history token is

```text
[state, commanded_action, completed_transition_reward, state_delta, done]
```

The reward is available only after the corresponding environment transition
has completed.  Current-query reward, future reward, hidden gain, alpha,
dataset-shard identity, and evaluation metadata are not context features.
Training uses no test rollouts and no test-time parameter updates.  The
Context-Q critic, value network, and posterior encoder are optimized from the
first offline update; the actor remains the native QGF flow-matching actor.

## Checkpoint-preserving Context-Q adapter finetuning

The adapter experiment asks whether causal context can improve a *trained*
native QGF critic without changing its initial action-gradient landscape.  It
is not a from-scratch comparison and it does not update the flow actor.

For each paired seed, it starts from the corresponding native
`params_500000.pkl` checkpoint and performs a fixed 30,000-update stage:

```bash
export OGBENCH_DATA_DIR=/data/ogbench
export NATIVE_CHECKPOINT=/path/to/native/qgf/run

./scripts/train_task3_context_q_adapter.sh 1
./scripts/train_task3_native_qv_continue.sh 1
```

The adapter command runs global offline steps `500001` through `530000`.  Its
trainable parameters are only:

```text
Q ContextInput projections
V ContextInput projection
causal PEARL posterior encoder
```

The flow actor, every copied native critic/value leaf, and Q/V output heads are
frozen.  Every `ContextInput` kernel is initialized to exact zero, so native
QGF and the new adapter model have the same policy, Q values, value estimates,
and `dQ/da` at initialization even with a nonempty ready history.  The target
critic updates only its newly added adapter leaves with `tau=0.005`; all native
target-Q leaves remain at their checkpoint values.

`train_task3_native_qv_continue.sh` is the required matched control: it starts
from the same checkpoint, freezes the same actor, and continues only native
Q/V updates for the same 30,000 updates.  The zero-adapter condition is the
pre-training transfer audit; it must be equivalent to frozen native QGF.

Both commands use the existing Task3 training shards (`cube-triple-play-100m-v0`,
excluding `*-val.npz`), action chunks of five, discount `0.999`, batch size
`1024`, and the causal 20-transition history.  The shard at resume is selected
from the global step, not reset to shard zero.  Batch starts are deterministically
derived from `(seed, global_step)`, ensuring paired native and adapter runs see
identical underlying transition windows regardless of logging or evaluation.

The scripts intentionally set `--eval_interval=0`: no online test rollout is
used while training or choosing the +30k checkpoint.  They save every 5k steps;
the primary report is the predeclared +30k checkpoint.  Existing MC, fixed-actor
closed-loop, and gradient diagnostics can load `context_qgf_adapter` checkpoints
directly; pass `EPOCH=530000` to the evaluation shell wrappers.

## MC-return calibration test

After both checkpoints exist, run:

```bash
NATIVE_CHECKPOINT=/path/to/native/run \
CONTEXT_CHECKPOINT=/path/to/context/run \
./scripts/evaluate_task3_context_q_mc.sh /path/to/task3_mc_result
```

The evaluator uses the same state, first action chunk, simulator reset/action
seeds, alpha, nominal gain `1.0`, and frozen native-QGF continuation for every
paired query.  It reports:

```text
Delta_MC = |Q_context - G_MC| - |Q_native - G_MC|
```

where `G_MC` is an empirical discounted continuation return.  A negative
`Delta_MC` indicates lower scalar Q error for Context-Q at the audited points;
it does not by itself establish a better QGF gradient or closed-loop success.
The output contains `result.json`, `queries.csv`, and `_SUCCESS`.
