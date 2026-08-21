#include "catch.hpp"
#include "pim_transaction.h"

TEST_CASE("PIM transaction decoder preserves legacy encoding", "[pim]") {
    using namespace dramsim3;

    auto launch = PimTransactionDecoder::Decode(0x3);
    REQUIRE(launch.kind == PimTransactionKind::START_COMPUTATION);
    REQUIRE(launch.launch_mask == 0x1);

    auto dataflow = PimTransactionDecoder::Decode(0x2c0860);
    REQUIRE(dataflow.kind == PimTransactionKind::LOAD_DATAFLOW_CONFIG);
    REQUIRE(dataflow.vcuts == 1);
    REQUIRE(dataflow.hcuts == 1);
    REQUIRE(dataflow.mcf == 2);
    REQUIRE(dataflow.ucf == 1);
    REQUIRE(dataflow.df == 0);
    REQUIRE(dataflow.m_tile_size == 2048);

    auto workload = PimTransactionDecoder::Decode(0x2000);
    REQUIRE(workload.kind == PimTransactionKind::LOAD_WORKLOAD_CONFIG);
    REQUIRE(workload.cut_no == 0);
    REQUIRE(workload.load_type == 0);
    REQUIRE(workload.dim_value == 64);
    REQUIRE(workload.base_row == 0);

    uint64_t boundary_address =
        (0x3fffffULL << 39) |
        (0x7fffffffULL << 7) |
        (2ULL << 5) |
        (15ULL << 1);
    auto boundary = PimTransactionDecoder::Decode(boundary_address);
    REQUIRE(boundary.kind == PimTransactionKind::LOAD_WORKLOAD_CONFIG);
    REQUIRE(boundary.cut_no == 15);
    REQUIRE(boundary.load_type == 2);
    REQUIRE(boundary.dim_value == 0x7fffffff);
    REQUIRE(boundary.base_row == 0x3fffff);

    uint64_t overlapping_marker = (3ULL << 5);
    REQUIRE(PimTransactionDecoder::Decode(overlapping_marker).kind ==
            PimTransactionKind::LOAD_DATAFLOW_CONFIG);
}
