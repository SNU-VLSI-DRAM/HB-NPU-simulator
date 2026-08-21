#include "catch.hpp"
#include "controller.h"
#include "pim_command_batch.h"
#include "pim_config.h"
#include "pim_partition_state.h"
#include "pim_transaction.h"
#include "timing.h"

namespace dramsim3 {
class ControllerTestPeer {
   public:
    static const std::vector<Command>& Weight(const Controller& controller) {
        return controller.rd_w_cmds_;
    }
    static const std::vector<Command>& Input(const Controller& controller) {
        return controller.rd_in_cmds_;
    }
    static const std::vector<int>& ReleaseTimes(const Controller& controller) {
        return controller.release_time;
    }
    static const std::vector<Command>& Output(const Controller& controller) {
        return controller.wr_cmds_;
    }
};
}  // namespace dramsim3

TEST_CASE("Controller PIM enqueue methods preserve order", "[pim]") {
    using namespace dramsim3;
    Config config("configs/HBM2_8Gb_x128_pim.ini", ".");
    Timing timing(config);
    Controller controller(0, config, timing);
    Command first(CommandType::PIM_ACTIVATE,
                  Address(0, 0, 0, 0, 1, 2), 3);
    Command second(CommandType::GH_READ,
                   Address(0, 0, 0, 1, 4, 5), 6);
    std::vector<Command> commands{first, second};

    controller.EnqueueWeightCommands(commands);
    controller.EnqueueInputCommands(commands, std::vector<int>{20, 21});
    controller.EnqueueOutputCommands(commands);

    REQUIRE(ControllerTestPeer::Weight(controller)[0].hex_addr == 3);
    REQUIRE(ControllerTestPeer::Weight(controller)[1].hex_addr == 6);
    REQUIRE(ControllerTestPeer::Input(controller)[0].hex_addr == 3);
    REQUIRE(ControllerTestPeer::Input(controller)[1].hex_addr == 6);
    REQUIRE(ControllerTestPeer::ReleaseTimes(controller) ==
            std::vector<int>{20, 21});
    REQUIRE(ControllerTestPeer::Output(controller)[0].hex_addr == 3);
    REQUIRE(ControllerTestPeer::Output(controller)[1].hex_addr == 6);
}

TEST_CASE("PIM command batch preserves order and release pairing", "[pim]") {
    using namespace dramsim3;
    PimCommandBatch batch;
    Command first(CommandType::PIM_ACTIVATE, Address(0, 0, 0, 0, 1, 2), 3);
    Command second(CommandType::LH_READ, Address(1, 0, 0, 1, 4, 5), 6);
    batch.AddInput(first, 10);
    batch.AddInput(second, 11);
    REQUIRE(batch.input_commands.size() == 2);
    REQUIRE(batch.input_commands[0].hex_addr == 3);
    REQUIRE(batch.input_commands[1].hex_addr == 6);
    REQUIRE(batch.input_release_times == std::vector<int>{10, 11});
}

TEST_CASE("PIM partition state defaults are independent", "[pim]") {
    using namespace dramsim3;
    std::vector<PimPartitionState> partitions(2);
    REQUIRE(partitions[0].base_row_in == 0);
    REQUIRE(partitions[0].m == 0);
    REQUIRE(partitions[0].in_pim == false);
    REQUIRE(partitions[0].out_cnt == -1);
    partitions[0].m = 17;
    partitions[0].in_act_placed = true;
    REQUIRE(partitions[1].m == 0);
    REQUIRE(partitions[1].in_act_placed == false);
}

TEST_CASE("PIM dataflow config preserves defaults and decoded values", "[pim]") {
    using namespace dramsim3;
    PimDataflowConfig config;
    REQUIRE(config.vcuts == -1);
    REQUIRE(config.hcuts == -1);
    REQUIRE(config.mcf == 1);
    REQUIRE(config.ucf == 1);
    REQUIRE(config.mc == 1);
    REQUIRE(config.df == -1);
    REQUIRE(config.m_tile_size == 0);

    config.Load(PimTransactionDecoder::Decode(0x2ec860));
    REQUIRE(config.vcuts == 1);
    REQUIRE(config.hcuts == 1);
    REQUIRE(config.mcf == 2);
    REQUIRE(config.ucf == 8);
    REQUIRE(config.mc == 16);
    REQUIRE(config.df == 1);
    REQUIRE(config.m_tile_size == 2048);
    REQUIRE(config.CutCount() == 1);
}

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
