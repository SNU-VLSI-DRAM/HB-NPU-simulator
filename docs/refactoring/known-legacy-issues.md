# Preserved HB-NPU Legacy Behavior

These observations were recorded during the initial behavior-preserving
refactor. Follow-up changes and their current status are identified below;
unresolved behavior changes still need dedicated expected results. See the
[integration overview](2026-09-hbnpu-updates.md) for the sequence of updates.

## Partially initialized transactions

- Location: `src/common.h`, `Transaction` constructors.
- Observation: constructors initialize different subsets of `is_write`,
  `is_pim`, address metadata, and scheduling metadata.
- Refactor treatment: preserve the constructor bodies exactly.
- Required future coverage: parse and construct every READ, WRITE, and PIM
  transaction form, then define deterministic defaults before changing them.

## Always-true output bank selection branch (simplified)

- Location: `src/dram_system.cc`, `JedecDRAMSystem::SchedulePimCommands`.
- Historical observation: the output `k_bound` condition contained `m == 1 || true`, so the
  alternative branch is unreachable.
- Initial refactor treatment: retained the condition and generated command count.
- Channel-command update: removed the unreachable alternative while retaining
  the effective mapping: one selected output bank for `df=1`, or `mc` selected
  banks for `df=0`. These are GH writes; `mc=2` interleaves two banks. Per-bank
  data-operation regression confirms the tested output addresses and order.

## Output completion for arbitrary matrix shapes

- Location: `src/dram_system.cc`, output geometry and completion conditions.
- Observation: a synthetic `df=1, M=64, K=128, N=256` workload reaches the
  `state.in_cnt == -1` assertion before input completion. The same assertion
  occurs in the saved pre-channel-command binary, so this is not introduced
  by gang commands or GH-write interleaving.
- Current treatment: preserve geometry; the 51 regression workloads pass but
  do not establish support for every encodable matrix shape.
- Required future coverage: define output geometry and completion behavior for
  partial/multiple N tiles before changing the formulas or suppressing assertions.

## Launch-mask validation semantics (removed)

- Location: `src/dram_system.cc`, `JedecDRAMSystem::ProcessPimTransaction`.
- Observation: the loop's empty statement and `else` association can reject a
  launch based on unselected cuts as well as selected cut configuration.
- Initial refactor treatment: preserved the loop, condition, and queue erase
  behavior.
- Single-array cleanup: removed masks and partition selection entirely. Launch
  now requires explicit dataflow initialization and nonzero M, K, and N.
  Unit tests cover pre-configuration queue retention and each missing dimension.
  See [the current format](../pim-trace-format.md); the historical multi-cut
  behavior is no longer an active compatibility requirement.

## Unused PIM readiness path and occupancy state

- Location: `src/dram_system.h` and `src/dram_system.cc`,
  `JedecDRAMSystem::GetReadyCommandPIM` and `bank_occupancy_`.
- Observation: the method and occupancy checks are not called by the active
  scheduling path.
- Refactor treatment: leave declarations, initialization, and implementation in
  place.
- Required future coverage: call-site audit across downstream users plus link
  and behavioral tests before removal.

## Hard-coded refresh lookahead constants

- Location: `src/refresh.cc`, `Refresh::pim_refresh_coming` and
  `Refresh::pim_refresh_coming2`.
- Observation: refresh lookahead uses literal `3` values alongside `tRAS`.
- Refactor treatment: preserve both comparisons.
- Required future coverage: commands and termination cycles immediately before,
  during, and after a refresh boundary.

## Repeated timing tables

- Location: `src/timing.cc`, `Timing::Timing`.
- Observation: multiple command-relation tables repeat PIM timing entries.
- Refactor treatment: no consolidation in this branch.
- Required future coverage: compare every affected same-bank, bank-group, rank,
  and channel timing relation before deduplication.
