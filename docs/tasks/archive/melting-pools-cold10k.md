# Melting fixed/rolling cold 10k

## Goal and acceptance

The user explicitly requests cold training to10000updates. Both fixed and
rolling arms start from fresh model/Adam states,with the same runtime seed43,
physical settings,initial-condition and time draws. Use the same8testcases.
Record models every250updates to examine late fluctuations and evaluate the
5000/10000states on the same trajectory. Historical checkpoint states are
comparison-only. The existing step-indexed LR schedule is retained.

## Status

Completed. Both cold10000runs and paired initialization/input audit exited0.
The checkpoint evaluator passed,including same-query final replay and exact
old5000IC/time/spatial draw-prefix identities. GPU is idle.

## Results

Same runtime seed43,FP32/Adam,B16/Nt8/K256,original step-based LR. Test is the
same8predeclared ellipsoids,24^3native diagnostic points and5times.
Values below are percentages,with one mean of per-input relative L2 per column.

| Mode | 5kpooled |5kterminal|10kpooled|10kterminal|9k..10kpooled median|
|---|---:|---:|---:|---:|---:|
|Fixed|0.234738|0.115385|0.120295|0.113172|0.120295|
|Rolling|0.413245|0.226426|0.100163|0.171115|0.143946|

The5kstates are the saved midpoints of this same cold10ktrajectory. At10k,
rolling improves pooledphase on7of8cases;fixed improves terminalphase on8of8.
Across9000/9250/9500/9750/10000both have large peaks at9250and change order.
Thus the final pooled ranking reverses the earlier5kresult,while the late
median favors fixed. The study does not identify a converged population ranking.

The cold10kmidpoint differs from the earlier independent5kseed43finalstate,
despite verified identical actual cold states and spatial/IC/time prefixes.
The exact floating execution mechanism was not isolated. Preserve this
ordinaryFP32optimizer-repeatability qualification instead of treating the
new midpoint as an exact reproduction of the old terminal checkpoint.

Same-card warm effective median16.2607/16.6838ms,activepeaks1.49375/1.49441GiB,
cold childprocess263.14/267.43s for fixed/rolling. Both loops start from fresh
model and Adam states; no trained state enters initialization. Models are
saved every250updates. Initializers match prior cold bytes and differ from
prior finalbytes.

## Artifacts and verification

Root /root/autodl-tmp/tf-logs/melting-pools-cold10k-20261111-01 contains
journal.json,paired-results.json,checkpoint-evaluation.json andconvergence.png.
The sibling .finish.log preserves checkpoint-replay values. Full models,
optimizer states,tracearrays and all250stepmodel snapshots are in the existing
example run root with suffixes -fixed_pooland-rolling_pool.

Actual fixed/rolling coldmodel/Adam byte equality and all10000IC/time/draw
identities pass. All6checkpoint evaluations perarm use the same8references;
final replay agrees within the existing2e-5physical field tolerance.
Sampling prefixes reconstruct the old5000actualarrays exactly. Config
validation and emitted250stepmodel snapshots are exercised by the normal
train.pyentry. No new reference solve or seed selection.

## Blocker and next action

None for this run. Further seeds,larger independent IC cohorts,stable-tail
optimizer trials or diagnosis of ordinaryGPUtrajectory repeatability are
separate follow-ups. No new training is queued by the current results.

## Task-end

Updated the example comparison and project memory with qualified results.
git diff --check passes. Memory files are PROGRESS,DECISIONSand this archive.
