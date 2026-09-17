# Task3 clean Context-Q experiment

This branch contains a from-scratch architecture comparison.  It does not
restore a native QGF checkpoint and does not run a 30k adapter stage.

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
