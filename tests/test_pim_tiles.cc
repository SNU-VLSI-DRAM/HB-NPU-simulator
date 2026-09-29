#include <stdexcept>
#include "catch.hpp"
#include "legacy_tile_scheduler.h"

using namespace dramsim3;
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
