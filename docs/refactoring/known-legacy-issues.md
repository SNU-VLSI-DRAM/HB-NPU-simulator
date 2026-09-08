# Preserved HB-NPU Legacy Behavior

These observations are intentionally unchanged by the behavior-preserving
refactor. Each needs a separate behavior-changing branch and dedicated expected
results.

## Partially initialized transactions

- Location: `src/common.h`, `Transaction` constructors.
- Observation: constructors initialize different subsets of `is_write`,
  `is_pim`, address metadata, and scheduling metadata.
- Refactor treatment: preserve the constructor bodies exactly.
- Required future coverage: parse and construct every READ, WRITE, and PIM
  transaction form, then define deterministic defaults before changing them.

## Always-true output bank selection branch

- Location: `src/dram_system.cc`, `JedecDRAMSystem::SchedulePimCommands`.
- Observation: the output `k_bound` condition contains `m == 1 || true`, so the
  alternative branch is unreachable.
- Refactor treatment: retain the condition and generated command count.
- Required future coverage: golden output commands for both `m == 1` and
  `m > 1` before deciding the intended bank selection.

## Launch-mask validation semantics

- Location: `src/dram_system.cc`, `JedecDRAMSystem::ProcessPimTransaction`.
- Observation: the loop's empty statement and `else` association can reject a
  launch based on unselected cuts as well as selected cut configuration.
- Refactor treatment: preserve the loop, condition, and queue erase behavior.
- Required future coverage: selected/unselected masks over configured and
  unconfigured cuts, including queue retention.

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
