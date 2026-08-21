# Task 4 Report: Mechanically Extract `JedecDRAMSystem::ClockTick`

## Pre-existing state assessment

The worktree started on the required refactor branch with uncommitted changes in
`src/dram_system.cc` and `src/dram_system.h` that had already split
`ClockTick` into the required helpers. There was no prior Task 4 report or
commit. The quick suite was run against that inherited state before any edit
and passed both tests. Eight untracked test-generated traces
(`dramsim3ch_8cmd.trace` through `dramsim3ch_15cmd.trace`) were present.

The inherited extraction had the required helper boundaries and orchestration.
Its only fidelity issue was normalization of the dispatch-loop bindings and
removal of an original comment. Those were restored without changing the
required const-reference helper interface. A trailing whitespace line was also
removed.

## Commands and results

1. `git status --short && git diff -- src/dram_system.h src/dram_system.cc && ctest --test-dir build --output-on-failure -L quick`
   - Inherited source diff and the eight untracked traces were confirmed.
   - `dramsim3_unit`: passed.
   - `hbnpu_quick_regression`: passed.
   - 2/2 tests passed.
2. `cmake --build build --target dramsim3 -j4`
   - Passed after the dispatch fidelity correction.
3. `cmake --build build --target dramsim3 -j4 && git diff --check`
   - Passed after removing trailing whitespace. The compiler emitted only the
     pre-existing signedness warnings in the mechanically moved legacy code.
4. `cmake --build build -j4 && ctest --test-dir build --output-on-failure -L quick`
   - Full build passed.
   - `dramsim3_unit`: passed.
   - `hbnpu_quick_regression`: passed.
   - 2/2 tests passed.
5. `git restore -- dramsim3ch_0cmd.trace dramsim3ch_1cmd.trace dramsim3ch_2cmd.trace dramsim3ch_3cmd.trace dramsim3ch_4cmd.trace dramsim3ch_5cmd.trace dramsim3ch_6cmd.trace dramsim3ch_7cmd.trace`
   - Restored tracked trace fixtures rewritten as a test side effect; no golden
     content was updated.
6. `rm dramsim3ch_8cmd.trace dramsim3ch_9cmd.trace dramsim3ch_10cmd.trace dramsim3ch_11cmd.trace dramsim3ch_12cmd.trace dramsim3ch_13cmd.trace dramsim3ch_14cmd.trace dramsim3ch_15cmd.trace`
   - Removed exactly the specified untracked generated traces.
7. `rg -n --glob 'src/**' --glob 'tests/**' '(->|\\.)(vcuts|hcuts|mcf|ucf|mc|df|vcuts_next|hcuts_next|M_tile_size|stride|kernel_size|base_rows_in|base_rows_w|base_rows_out|M|N|K|M_it|N_it|K_tile_it|M_out_it|N_out_tile_it|in_pim|iw_status|in_act_placed|w_act_placed|out_act_placed|output_valid|in_cnt|out_cnt|vpu_cnt|bank_occupancy_|pim_trans_queue_|pim_trans_queue_depth_)\\b' . || true`
   - No external `JedecDRAMSystem` state access was found. The sole match was
     the distinct controller field `ctrls_[i]->in_pim`.

## Behavior-preservation self-review

- `ClockTick` follows the mandated phase order exactly: completed
  transactions, refresh-window check, one PIM transaction, refresh-blocked
  check, scheduling, controller ticks, clock increment, then epoch stats.
- The private declarations exactly match the required helper signatures.
  `GetReadyCommandPIM` remains public and unchanged.
- Transaction completion, refresh handling, PIM queue handling, scheduler
  loop nesting, assertions, `AbruptExit`, state mutations, and controller
  tick ordering were moved without logic changes.
- Dispatch preserves weight/input/output order and computes `release_time_`
  from `clk_` at the original point before queue insertion completes.
- No state representation was changed, no goldens were updated, and no
  Qwen3/Mistral/GQA work was included.

## Files changed

- `src/dram_system.h`
- `src/dram_system.cc`
- `.superpowers/sdd/2026-08-14-behavior-preserving-hbnpu-refactor/task-4-report.md`

## Concerns

None. The compiler reports existing signed/unsigned comparison warnings in
legacy scheduler code; they were intentionally left unchanged.
