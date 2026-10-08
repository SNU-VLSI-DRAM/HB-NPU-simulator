#include <stdexcept>
#include <algorithm>
#include "catch.hpp"
#include "hbnpu/legacy_tile_scheduler.h"
#include "hbnpu/pim_tile_executor.h"

using namespace dramsim3;
namespace dramsim3 {
// Inject only the counter-boundary states that would otherwise require long
// command sequences; descriptor ownership still goes through the real API.
class PimTileExecutorTestPeer {
 public:
    static void Prime(PimTileExecutor& executor, const PimExecutionState& state) {
        LegacyTileScheduler scheduler(executor.context_, executor.config_.channels);
        PimComputeBlock compute;
        PimOutputBlock output;
        scheduler.NextCompute(compute);
        scheduler.NextOutput(output);
        executor.LoadCompute(compute);
        executor.LoadOutput(output);
        executor.state_ = state;
    }
};
}
namespace {
PimKernelContext MakeContext(int m, int n, int k, int df) {
    PimExecutionState state;
    state.m = m; state.n = n; state.k = k;
    PimDataflowConfig dataflow;
    dataflow.Load(PimTransactionDecoder::Decode(df == 0 ? 0x2c0e : 0x2ece));
    return PimKernelContext::Capture(dataflow, state);
}
}

TEST_CASE("Compute descriptors traverse K then N then M", "[pim][tile]") {
    auto c = MakeContext(65, 256, 256, 0);
    c.dataflow.m_tile_size = 64;
    LegacyTileScheduler schedule(c, 8);
    PimComputeBlock block;
    int count = 0;
    for (int m : {0, 64}) for (int n : {0, 128}) for (int k : {0, 128}) {
        REQUIRE(schedule.NextCompute(block));
        REQUIRE(block.m_origin == m);
        REQUIRE(block.n_origin == n);
        REQUIRE(block.k_origin == k);
        REQUIRE(block.m_extent == (m == 0 ? 64 : 1));
        REQUIRE(block.n_extent == 128);
        REQUIRE(block.k_extent == 128);
        REQUIRE(block.first_k == (k == 0));
        REQUIRE(block.final_k == (k == 128));
        REQUIRE(block.last_block == (++count == 8));
    }
    REQUIRE_FALSE(schedule.NextCompute(block));
    REQUIRE(block.m_origin == 64);
    REQUIRE(block.k_origin == 128);
}

TEST_CASE("Descriptor tails preserve launch geometry", "[pim][tile]") {
    auto c = MakeContext(3, 129, 130, 0);
    LegacyTileScheduler schedule(c, 8);
    PimComputeBlock block;
    REQUIRE(schedule.NextCompute(block));
    REQUIRE(block.k_extent == 128);
    REQUIRE(schedule.NextCompute(block));
    REQUIRE(block.k_extent == 2);
    REQUIRE(block.final_k);
    REQUIRE(schedule.NextCompute(block));
    REQUIRE(block.n_origin == 128);
    REQUIRE(block.n_extent == 1);
    REQUIRE(block.m_extent == 3);
    REQUIRE(schedule.NextCompute(block));
    REQUIRE(block.last_block);
    REQUIRE_FALSE(schedule.NextCompute(block));
}

TEST_CASE("Output descriptors have independent dataflow-specific traversal", "[pim][tile]") {
    auto c = MakeContext(65, 256, 256, 0);
    c.dataflow.m_tile_size = 64;
    SECTION("GH output traverses N then M") {
        LegacyTileScheduler schedule(c, 8);
        PimOutputBlock out;
        PimComputeBlock block;
        int count = 0;
        for (int m : {0, 64}) for (int n : {0, 1}) {
            REQUIRE(schedule.NextOutput(out));
            REQUIRE(out.m_origin == m);
            REQUIRE(out.n_tile == n);
            REQUIRE(out.m_extent == (m == 0 ? 64 : 1));
            REQUIRE(out.last_output == (++count == 4));
            REQUIRE(schedule.NextCompute(block));
            REQUIRE(block.m_origin == 0);
        }
        REQUIRE_FALSE(schedule.NextOutput(out));
        REQUIRE(out.m_origin == 64);
    }
    SECTION("LH output uses transformed M and fixed N") {
        c = MakeContext(256, 256, 256, 1);
        c.dataflow.m_tile_size = 128;
        LegacyTileScheduler schedule(c, 8);
        PimOutputBlock out;
        for (int m : {0, 2}) {
            REQUIRE(schedule.NextOutput(out));
            REQUIRE(out.m_origin == m);
            REQUIRE(out.n_tile == 0);
            REQUIRE(out.m_extent == 2);
            REQUIRE(out.last_output == (m == 2));
        }
        REQUIRE_FALSE(schedule.NextOutput(out));
    }
}

TEST_CASE("Captured context and scheduler do not alias launch staging", "[pim][tile]") {
    PimExecutionState state;
    state.m = 64; state.n = 128; state.k = 128;
    state.base_row_in = 17; state.base_row_weight = 31; state.base_row_output = 49;
    PimDataflowConfig dataflow;
    dataflow.Load(PimTransactionDecoder::Decode(0x2c0e));
    auto c = PimKernelContext::Capture(dataflow, state);
    LegacyTileScheduler schedule(c, 8);
    state.m = 1; dataflow.mc = 1; c.m = 2;
    PimComputeBlock block;
    REQUIRE(schedule.NextCompute(block));
    REQUIRE(block.m_extent == 64);
    REQUIRE(c.dataflow.mc == 2);
    REQUIRE(c.base_row_in == 17);
    REQUIRE(c.base_row_weight == 31);
    REQUIRE(c.base_row_output == 49);
}

TEST_CASE("Invalid internal kernel geometry is rejected", "[pim][tile]") {
    auto c = MakeContext(64, 128, 128, 0);
    int channels = 8;
    SECTION("unconfigured") { c.dataflow.configured = false; }
    SECTION("M") { c.m = 0; }
    SECTION("N") { c.n = 0; }
    SECTION("K") { c.k = 0; }
    SECTION("M tile") { c.dataflow.m_tile_size = 0; }
    SECTION("channels") { channels = 0; }
    SECTION("mc") { c.dataflow.mc = 0; }
    SECTION("dataflow") { c.dataflow.df = 2; }
    SECTION("LH output tile") { c.dataflow.df = 1; c.dataflow.m_tile_size = 64; }
    REQUIRE_THROWS_AS(LegacyTileScheduler(c, channels), std::invalid_argument);
}

TEST_CASE("Executor retires input vectors before starting same-tick output delay", "[pim][tile]") {
    Config hardware("configs/HBM2_8Gb_x128_pim.ini", ".");
    int df = 0, expected = 4;
    SECTION("GH") {}
    SECTION("LH") { df = 1; expected = 23; }
    auto context = MakeContext(64, 256, 128, df);
    PimTileExecutor executor(hardware, context);
    PimExecutionState state;
    state.m = 64; state.n = 256; state.k = 128;
    state.in_pim = true; state.iw_status = 2; state.input_pending = true;
    PimTileExecutorTestPeer::Prime(executor, state);
    auto result = executor.Tick(100, PimTickFeedback(), PimDescriptorSource());
    state = executor.Snapshot();
    REQUIRE(state.m_it == 1);
    REQUIRE(state.out_cnt == expected);
    REQUIRE(std::count(result.events.begin(), result.events.end(), PimProgress::InputVectorIssued) == 1);
}

TEST_CASE("Executor distinguishes pending data and refresh from data retirement", "[pim][tile]") {
    Config hardware("configs/HBM2_8Gb_x128_pim.ini", ".");
    PimTileExecutor executor(hardware, MakeContext(64, 256, 128, 0));
    PimExecutionState state;
    state.m = 64; state.n = 256; state.k = 128;
    state.in_pim = true; state.iw_status = 2; state.input_pending = true;
    PimTickFeedback feedback;
    bool retired = false;
    SECTION("one or more channels still pending") { feedback.input_pending = true; }
    SECTION("refresh freezes execution") { feedback.in_refresh = true; }
    SECTION("refresh lookahead permits retirement but not enqueue") {
        feedback.wait_refresh = true; retired = true;
    }
    PimTileExecutorTestPeer::Prime(executor, state);
    auto result = executor.Tick(100, feedback, PimDescriptorSource());
    state = executor.Snapshot();
    REQUIRE(state.m_it == (retired ? 1 : 0));
    REQUIRE(result.batch.operations.empty());
    REQUIRE(std::count(result.events.begin(), result.events.end(), PimProgress::InputVectorIssued) == (retired ? 1 : 0));
}

TEST_CASE("Output uses entry phase and releases counter exactly once", "[pim][tile]") {
    Config hardware("configs/HBM2_8Gb_x128_pim.ini", ".");
    auto context = MakeContext(64, 256, 128, 0);
    PimTileExecutor executor(hardware, context);
    PimExecutionState state;
    state.m = 64; state.n = 256; state.k = 128;
    state.m_it = 63; state.iw_status = 2; state.in_pim = true;
    state.input_pending = true; state.out_cnt = 0;
    PimTileExecutorTestPeer::Prime(executor, state);
    auto result = executor.Tick(100, PimTickFeedback(), PimDescriptorSource());
    state = executor.Snapshot();
    REQUIRE(state.iw_status == 3);
    REQUIRE(state.out_cnt == -1);
    REQUIRE(state.output_valid == 1);
    REQUIRE(result.batch.operations.empty());
    REQUIRE(std::count(result.events.begin(), result.events.end(), PimProgress::OutputStageReleased) == 1);
    result = executor.Tick(101, PimTickFeedback(), PimDescriptorSource());
    REQUIRE_FALSE(result.batch.operations.empty());
    REQUIRE(std::count(result.events.begin(), result.events.end(), PimProgress::OutputStageReleased) == 0);
}

TEST_CASE("One supplied block cannot start the next matrix block", "[pim][tile]") {
    Config hardware("configs/HBM2_8Gb_x128_pim.ini", ".");
    auto c = MakeContext(3, 128, 256, 0);
    LegacyTileScheduler schedule(c, 8);
    PimTileExecutor executor(hardware, c);
    PimDescriptorSource source;
    int requests = 0;
    source.next_compute = [&](PimComputeBlock& block) {
        return ++requests == 1 && schedule.NextCompute(block);
    };
    int exhausted = 0;
    for (int clk = 0; clk < 300; ++clk) {
        auto result = executor.Tick(clk, PimTickFeedback(), source);
        exhausted += std::count(result.events.begin(), result.events.end(), PimProgress::BlockStreamingExhausted);
        REQUIRE_FALSE(result.finished);
        if (clk >= 200) REQUIRE(result.batch.operations.empty());
    }
    REQUIRE(exhausted == 1);
    REQUIRE(requests > 1);
    REQUIRE(executor.CanAcceptCompute());
}

TEST_CASE("Active descriptors are owned and cannot be replaced", "[pim][tile]") {
    Config hardware("configs/HBM2_8Gb_x128_pim.ini", ".");
    auto c = MakeContext(64, 256, 128, 0);
    LegacyTileScheduler schedule(c, 8);
    PimTileExecutor executor(hardware, c);
    PimComputeBlock block;
    PimOutputBlock output;
    REQUIRE(schedule.NextCompute(block));
    REQUIRE(schedule.NextOutput(output));
    executor.LoadCompute(block);
    executor.LoadOutput(output);
    REQUIRE_FALSE(executor.CanAcceptCompute());
    REQUIRE_FALSE(executor.CanAcceptOutput());
    REQUIRE_THROWS_AS(executor.LoadCompute(block), std::logic_error);
    REQUIRE_THROWS_AS(executor.LoadOutput(output), std::logic_error);
    block.m_origin = 99; output.m_origin = 99;
    REQUIRE(executor.Snapshot().m_it == 0);
    REQUIRE(executor.Snapshot().m_out_it == 0);
}

TEST_CASE("Missing output descriptor stalls completion without losing availability", "[pim][tile]") {
    Config hardware("configs/HBM2_8Gb_x128_pim.ini", ".");
    auto c = MakeContext(3, 128, 128, 0);
    LegacyTileScheduler schedule(c, 8);
    PimTileExecutor executor(hardware, c);
    PimDescriptorSource source;
    source.next_compute = [&](PimComputeBlock& b) { return schedule.NextCompute(b); };
    for (int clk = 0; clk < 200; ++clk) {
        auto r = executor.Tick(clk, PimTickFeedback(), source);
        REQUIRE_FALSE(r.finished);
        for (const auto& op : r.batch.operations) REQUIRE(op.first != PimSource::OUTPUT);
    }
    REQUIRE(executor.Snapshot().output_valid == 1);
    REQUIRE(executor.Snapshot().in_cnt == -1);
    source.next_output = [&](PimOutputBlock& b) { return schedule.NextOutput(b); };
    bool finished = false;
    for (int clk = 200; clk < 210; ++clk) {
        auto r = executor.Tick(clk, PimTickFeedback(), source);
        if (r.finished) { finished = true; break; }
    }
    REQUIRE(finished);
    REQUIRE_FALSE(executor.Snapshot().in_pim);
    REQUIRE(executor.Tick(211, PimTickFeedback(), source).batch.operations.empty());
}

TEST_CASE("Legacy diagnostics project exhausted cursors without advancing selection", "[pim][tile]") {
    auto c = MakeContext(65, 256, 256, 0);
    c.dataflow.m_tile_size = 64;
    LegacyTileScheduler schedule(c, 8);
    PimComputeBlock b;
    PimOutputBlock o;
    PimExecutionState state;
    REQUIRE(schedule.NextCompute(b));
    REQUIRE(schedule.NextOutput(o));
    schedule.ProjectProgress(state, true, true);
    REQUIRE(state.k_tile_it == 1);
    REQUIRE(state.n_it == 0);
    REQUIRE(state.m_it == 0);
    REQUIRE(state.n_out_tile_it == 1);
    REQUIRE(state.m_out_it == 0);
    REQUIRE(schedule.NextCompute(b));
    REQUIRE(b.k_origin == 128);
    while (schedule.NextCompute(b)) {}
    while (schedule.NextOutput(o)) {}
    schedule.ProjectProgress(state, true, true);
    REQUIRE(state.k_tile_it == 0);
    REQUIRE(state.n_it == 0);
    REQUIRE(state.m_it == 128);
    REQUIRE(state.n_out_tile_it == 0);
    REQUIRE(state.m_out_it == 128);
}

TEST_CASE("Executor rejects malformed block geometry without taking ownership", "[pim][tile]") {
    Config hardware("configs/HBM2_8Gb_x128_pim.ini", ".");
    auto c = MakeContext(64, 256, 256, 0);
    LegacyTileScheduler schedule(c, 8);
    PimTileExecutor executor(hardware, c);
    PimComputeBlock b;
    REQUIRE(schedule.NextCompute(b));
    SECTION("zero extent") { b.m_extent = 0; }
    SECTION("out of bounds") { b.n_origin = 256; }
    SECTION("unaligned K") { b.k_origin = 1; }
    SECTION("wrong first flag") { b.first_k = false; }
    SECTION("wrong final flag") { b.final_k = true; }
    SECTION("premature last block") { b.last_block = true; }
    REQUIRE_THROWS_AS(executor.LoadCompute(b), std::invalid_argument);
    REQUIRE(executor.CanAcceptCompute());
}

TEST_CASE("Executor rejects malformed output geometry without taking ownership", "[pim][tile]") {
    Config hardware("configs/HBM2_8Gb_x128_pim.ini", ".");
    auto c = MakeContext(64, 256, 128, 0);
    LegacyTileScheduler schedule(c, 8);
    PimTileExecutor executor(hardware, c);
    PimOutputBlock b;
    REQUIRE(schedule.NextOutput(b));
    SECTION("negative origin") { b.m_origin = -1; }
    SECTION("zero extent") { b.m_extent = 0; }
    SECTION("out of bounds N") { b.n_tile = 2; }
    SECTION("premature last output") { b.last_output = true; }
    REQUIRE_THROWS_AS(executor.LoadOutput(b), std::invalid_argument);
    REQUIRE(executor.CanAcceptOutput());
}
