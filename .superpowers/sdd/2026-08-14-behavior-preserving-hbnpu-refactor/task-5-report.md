# Task 5 Report: Mechanically Extract Controller PIM Scheduling

## Pre-existing state assessment

The worktree started on `refactor/behavior-preserving-hbnpu` at the required
base commit `f902e73954af53f78722c75a4a243075f64b2833`. It already contained
uncommitted changes in `src/controller.cc` and `src/controller.h`, but no Task 5
report or commit. Tracked root command traces `dramsim3ch_0cmd.trace` through
`dramsim3ch_7cmd.trace` had been rewritten, and generated root traces
`dramsim3ch_8cmd.trace` through `dramsim3ch_15cmd.trace` were untracked.

The inherited controller changes were audited against both the Task 5 brief and
the original source at `f902e739`. They already implemented the requested pure
mechanical extraction exactly, so no corrective source edit was needed. The
four helper declarations had the exact requested signatures and were private.
The three loop bodies were moved without changing their statements or control
flow, and `ClockTick` retained the original branch structure and dispatch
priority.

## Commands and results

1. `ctest --test-dir build --output-on-failure -L quick`
   - Run against the inherited extraction before any source or report edit.
   - `dramsim3_unit`: passed.
   - `hbnpu_quick_regression`: passed.
   - 2/2 tests passed.
2. `git show f902e739:src/controller.cc | nl -ba | sed -n '65,225p'` and
   `nl -ba src/controller.cc | sed -n '65,235p'`
   - Compared the original inline scheduling blocks with all extracted helpers.
   - Confirmed literal preservation of loop statements and ordering.
3. `rg -n "PimQueuesEmpty|ScheduleWeightPimCommands|ScheduleInputPimCommands|ScheduleOutputPimCommands|wr_multitenant|release_time\\[i\\]|IsRefreshWaiting" src/controller.cc src/controller.h`
   - Confirmed the exact helper interfaces, weight/input/output call order,
     refresh guards, release-time indexing, and multitenant break placement.
4. `git diff --check`
   - Passed with no whitespace errors.
5. `cmake --build build -j4`
   - Full build passed; `dramsim3` and `dramsim3main` were built successfully.
6. `ctest --test-dir build --output-on-failure -L quick`
   - Post-build verification passed.
   - `dramsim3_unit`: passed.
   - `hbnpu_quick_regression`: passed.
   - 2/2 tests passed.
7. `git restore -- dramsim3ch_0cmd.trace ... dramsim3ch_7cmd.trace`
   - Restored only the eight tracked root traces rewritten by testing; no golden
     content was updated.
8. `rm dramsim3ch_8cmd.trace ... dramsim3ch_15cmd.trace`
   - Removed exactly the eight specified untracked generated traces.

## Behavior-preservation self-review

- `PimQueuesEmpty() const` exactly represents the former conjunction of
  `rd_w_cmds_`, `rd_in_cmds_`, and `wr_cmds_` emptiness checks.
- `ClockTick` still gives a refresh command precedence and enters the PIM branch
  under the same condition as before.
- PIM scheduling calls remain in weight, input, then output order.
- Weight scheduling preserves iterator increment/erase order, the
  `IsRefreshWaiting()` activation gate, command readiness checks, and command
  issue/erase behavior.
- Input scheduling preserves `i` and `j` initialization and advancement,
  `release_time[i]` lookup, synchronized release-time erase, refresh gating,
  and iterator behavior.
- Output scheduling preserves readiness handling, refresh gating, both erase
  paths, and the `if (wr_multitenant) break;` position after issue and erase.
- Existing unused locals, comments, formatting, public queue/state
  representation, and simulator behavior were intentionally left unchanged.
- No legacy cleanup, golden updates, Qwen, Mistral, or GQA work was included.

## Files changed

- `src/controller.h`
- `src/controller.cc`
- `.superpowers/sdd/2026-08-14-behavior-preserving-hbnpu-refactor/task-5-report.md`

## Concerns

None.
