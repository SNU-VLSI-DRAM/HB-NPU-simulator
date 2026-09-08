#include "catch.hpp"
#include "controller.h"
#include "dram_system.h"
#include "pim_command_batch.h"
#include "pim_config.h"
#include "pim_execution_state.h"
#include "pim_transaction.h"
#include "timing.h"

namespace {
void dummy_pim_callback(uint64_t) {}
}

namespace dramsim3 {
class ControllerTestPeer {
   public:
    static void OpenBank(Controller& controller, int group, int bank, int rank = 0) {
        controller.channel_state_.UpdateTimingAndStates(
            Command(CommandType::PIM_ACTIVATE,Address(controller.channel_id_,rank,group,bank,0,0),0),0);
    }
    static void RequestRefresh(Controller& controller) {
        controller.channel_state_.RankNeedRefresh(0,true);
    }
    static bool IsOpen(const Controller& controller,int group,int bank) {
        return controller.channel_state_.IsRowOpen(0,group,bank);
    }
    static bool IsSelfRefreshing(const Controller& controller, int rank) {
        return controller.channel_state_.IsRankSelfRefreshing(rank);
    }
    static void PrepareOpenBanks(Controller& controller) {
        // Start after activation latency, with two banks ready for row hits.
        for (int group : {0, 2}) {
            Command activate(CommandType::PIM_ACTIVATE,
                             Address(controller.channel_id_, 0, group, 0, 0, 0), 0);
            controller.channel_state_.UpdateTimingAndStates(activate, 0);
        }
        controller.clk_ = 100;
        for (int cycle=0; cycle<100; ++cycle) controller.cmd_queue_.ClockTick();
        controller.in_pim = true;
    }
    static std::vector<Command> Flatten(const std::vector<PimOperation>& operations) {
        std::vector<Command> result;
        for (const auto& operation : operations)
            result.insert(result.end(),operation.commands.begin(),operation.commands.end());
        return result;
    }
    static std::vector<Command> Weight(const Controller& controller) {
        return Flatten(controller.rd_w_cmds_);
    }
    static std::vector<Command> Input(const Controller& controller) {
        return Flatten(controller.rd_in_cmds_);
    }
    static const std::vector<int>& ReleaseTimes(const Controller& controller) {
        return controller.release_time;
    }
    static std::vector<Command> Output(const Controller& controller) {
        return Flatten(controller.wr_cmds_);
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

TEST_CASE("GH reads interleave banks at one command per tick", "[pim][gh-timing]") {
    using namespace dramsim3;
    Config config("configs/HBM2_8Gb_x128_pim.ini", ".");
    Timing timing(config);
    Controller controller(0, config, timing);
    ControllerTestPeer::PrepareOpenBanks(controller);
    std::vector<Command> commands;
    for (int column : {0, 1}) {
        for (int group : {0, 2}) {
            commands.emplace_back(CommandType::GH_READ,
                                  Address(0, 0, group, 0, 0, column), 0);
        }
    }
    controller.EnqueueInputCommands(commands, {100, 100, 100, 100});
    for (size_t remaining : {3, 2, 1, 0}) {
        controller.ClockTick();
        REQUIRE(ControllerTestPeer::Input(controller).size() == remaining);
    }
}

TEST_CASE("GH output writes interleave at one command per tick", "[pim][gh-timing]") {
    using namespace dramsim3;
    Config config("configs/HBM2_8Gb_x128_pim.ini", ".");
    Timing timing(config);
    Controller controller(0, config, timing);
    ControllerTestPeer::PrepareOpenBanks(controller);
    std::vector<Command> writes;
    for (int column : {0, 1})
        for (int group : {0, 2})
            writes.emplace_back(CommandType::PIM_WRITE,
                                Address(0, 0, group, 0, 0, column), 0);
    controller.EnqueueOutputCommands(writes);
    for (size_t remaining : {3, 2, 1, 0}) {
        controller.ClockTick();
        REQUIRE(ControllerTestPeer::Output(controller).size() == remaining);
    }
}

TEST_CASE("GH reads and output writes share the channel slot", "[pim][gh-timing]") {
    using namespace dramsim3;
    Config config("configs/HBM2_8Gb_x128_pim.ini", ".");
    Timing timing(config);
    Controller controller(0, config, timing);
    ControllerTestPeer::PrepareOpenBanks(controller);
    CommandType write = CommandType::PIM_WRITE;
    SECTION("write") {}
    SECTION("write with precharge") { write = CommandType::PIM_WRITE_PRECHARGE; }
    controller.EnqueueWeightCommands({Command(CommandType::GH_READ, Address(0,0,0,0,0,0),0)});
    controller.EnqueueOutputCommands({Command(write, Address(0,0,2,0,0,0),0)});
    controller.ClockTick();
    REQUIRE(ControllerTestPeer::Weight(controller).empty());
    REQUIRE(ControllerTestPeer::Output(controller).size() == 1);
    controller.ClockTick();
    REQUIRE(ControllerTestPeer::Output(controller).empty());
}

TEST_CASE("GH writes retain bank cadence even for a one-cycle burst", "[pim][gh-timing]") {
    using namespace dramsim3;
    Config config("configs/HBM2_8Gb_x128_pim.ini", ".");
    config.burst_cycle = 1;
    config.tCCD_S = 1;
    config.tCCD_L = 2;
    Timing timing(config);
    Controller controller(0,config,timing);
    ControllerTestPeer::PrepareOpenBanks(controller);
    std::vector<Command> writes;
    for (int group : {0,2})
        for (int col : {0,1}) writes.emplace_back(CommandType::PIM_WRITE,Address(0,0,group,0,0,col),0);
    controller.EnqueueOutputCommands(writes);
    controller.ClockTick();
    controller.ClockTick();
    auto remaining = ControllerTestPeer::Output(controller);
    REQUIRE(remaining.size()==2);
    REQUIRE(remaining.front().Bankgroup()==0);
    REQUIRE(remaining.front().Column()==1);
}

TEST_CASE("LH writes do not consume the global transfer bandwidth slot", "[pim][gh-timing]") {
    using namespace dramsim3;
    Config config("configs/HBM2_8Gb_x128_pim.ini", ".");
    config.ranks = 2;
    config.tCCD_S = 4;
    Timing timing(config);
    Controller controller(0,config,timing);
    ControllerTestPeer::PrepareOpenBanks(controller);
    std::vector<Command> writes;
    for (int bank=0; bank<16; ++bank) {
        if (!(bank==0 || bank==8)) ControllerTestPeer::OpenBank(controller,bank/4,bank%4);
        writes.emplace_back(CommandType::PIM_WRITE,Address(0,0,bank/4,bank%4,0,0),0);
    }
    ControllerTestPeer::OpenBank(controller,0,0,1);
    controller.EnqueuePimOperation(PimSource::WEIGHT,PimOperation(PimOpcode::LH_WRITE,writes));
    controller.EnqueueOutputCommands({Command(CommandType::GH_READ,Address(0,1,0,0,0,0),0)});
    controller.ClockTick();
    REQUIRE_FALSE(controller.HasPendingPim(PimSource::WEIGHT));
    REQUIRE(controller.HasPendingPim(PimSource::OUTPUT));
    controller.ClockTick();
    REQUIRE_FALSE(controller.HasPendingPim(PimSource::OUTPUT));
}

TEST_CASE("Pending PIM data survives refresh preparation and reactivation", "[pim][gang]") {
    using namespace dramsim3;
    Config config("configs/HBM2_8Gb_x128_pim.ini", ".");
    Timing timing(config);
    Controller controller(0,config,timing);
    ControllerTestPeer::PrepareOpenBanks(controller);
    controller.EnqueueOutputCommands({Command(CommandType::PIM_WRITE,Address(0,0,0,0,0,0),0)});
    ControllerTestPeer::RequestRefresh(controller);
    controller.ClockTick();
    REQUIRE(controller.HasPendingPim(PimSource::OUTPUT));
    for (int cycle=0; cycle<config.tRFC+200; ++cycle) controller.ClockTick();
    REQUIRE_FALSE(controller.HasPendingPim(PimSource::OUTPUT));
}

TEST_CASE("All PIM prerequisite queues share one issue slot and retain data", "[pim][gang]") {
    using namespace dramsim3;
    Config config("configs/HBM2_8Gb_x128_pim.ini", ".");
    Timing timing(config);
    Controller controller(0,config,timing);
    controller.EnqueueWeightCommands({Command(CommandType::GH_READ,Address(0,0,0,0,0,0),0)});
    controller.EnqueueInputCommands({Command(CommandType::GH_READ,Address(0,0,1,0,0,0),0)},{0});
    controller.EnqueueOutputCommands({Command(CommandType::PIM_WRITE,Address(0,0,2,0,0,0),0)});
    for (int tick=0; tick<3; ++tick) {
        controller.ClockTick();
        for (int group=0; group<3; ++group)
            REQUIRE(ControllerTestPeer::IsOpen(controller,group,0)==(group<=tick));
        REQUIRE(controller.HasPendingPim(PimSource::WEIGHT));
        REQUIRE(controller.HasPendingPim(PimSource::INPUT));
        REQUIRE(controller.HasPendingPim(PimSource::OUTPUT));
    }
    for (int tick=0; tick<100; ++tick) controller.ClockTick();
    REQUIRE_FALSE(controller.HasPendingPim(PimSource::WEIGHT));
    REQUIRE_FALSE(controller.HasPendingPim(PimSource::INPUT));
    REQUIRE_FALSE(controller.HasPendingPim(PimSource::OUTPUT));
}

TEST_CASE("Kernel completion waits for issued data through refresh", "[pim][gang]") {
    using namespace dramsim3;
    Config config("configs/HBM2_8Gb_x128_pim.ini", ".");
    config.tREFI=300;
    config.tRFC=30;
    JedecDRAMSystem system(config,".",dummy_pim_callback,dummy_pim_callback);
    for (uint64_t address : {uint64_t(0x2ece),uint64_t(0x200),uint64_t(0x402),uint64_t(0x404),uint64_t(1)})
        REQUIRE(system.AddTransaction(address));
    for (int cycle=0; cycle<10000 && !system.turn_off; ++cycle) system.ClockTick();
    REQUIRE(system.turn_off);
    REQUIRE_FALSE(system.execution_.input_pending);
    REQUIRE_FALSE(system.execution_.weight_pending);
    REQUIRE_FALSE(system.execution_.output_pending);
}

TEST_CASE("Release-blocked PIM work prevents self-refresh in its rank", "[pim][self-refresh]") {
    using namespace dramsim3;
    Config config("configs/HBM2_8Gb_x128_pim.ini", ".");
    config.enable_self_refresh = true;
    config.sref_threshold = 1;
    Timing timing(config);
    Controller controller(0, config, timing);
    PimSource source = PimSource::INPUT;
    SECTION("weight queue") { source = PimSource::WEIGHT; }
    SECTION("input queue") {}
    SECTION("output queue") { source = PimSource::OUTPUT; }
    controller.EnqueuePimOperation(source,
        PimOperation(Command(CommandType::GH_READ, Address(0,0,0,0,0,0),0),10));
    controller.ClockTick();
    REQUIRE_FALSE(ControllerTestPeer::IsSelfRefreshing(controller,0));
    REQUIRE_FALSE(ControllerTestPeer::IsOpen(controller,0,0));
    REQUIRE(controller.HasPendingPim(source));
    for (int cycle=1; cycle<10+config.tRCDRD+1; ++cycle) controller.ClockTick();
    REQUIRE_FALSE(controller.HasPendingPim(source));
}

TEST_CASE("PIM arrival wakes a sleeping rank and honors exit timing", "[pim][self-refresh]") {
    using namespace dramsim3;
    Config config("configs/HBM2_8Gb_x128_pim.ini", ".");
    config.enable_self_refresh = true;
    config.sref_threshold = 1;
    config.tCKESR = 3;
    config.tXS = 5;
    config.tRCDRD = 2;
    Timing timing(config);
    Controller controller(0, config, timing);
    controller.ClockTick(); // SREF_ENTER at cycle 0, as in normal-only operation.
    REQUIRE(ControllerTestPeer::IsSelfRefreshing(controller,0));
    controller.EnqueueInputCommands({Command(CommandType::GH_READ,Address(0,0,0,0,0,0),0)},{0});
    for (int cycle=1; cycle<3; ++cycle) {
        controller.ClockTick();
        REQUIRE(ControllerTestPeer::IsSelfRefreshing(controller,0));
    }
    controller.ClockTick(); // SREF_EXIT at cycle 3 consumes this tick.
    REQUIRE_FALSE(ControllerTestPeer::IsSelfRefreshing(controller,0));
    REQUIRE_FALSE(ControllerTestPeer::IsOpen(controller,0,0));
    REQUIRE(controller.HasPendingPim(PimSource::INPUT));
    for (int cycle=4; cycle<8; ++cycle) {
        controller.ClockTick();
        REQUIRE_FALSE(ControllerTestPeer::IsOpen(controller,0,0));
    }
    controller.ClockTick(); // ACT at 3+tXS=8.
    REQUIRE(ControllerTestPeer::IsOpen(controller,0,0));
    REQUIRE(controller.HasPendingPim(PimSource::INPUT));
    controller.ClockTick();
    REQUIRE(controller.HasPendingPim(PimSource::INPUT));
    controller.ClockTick(); // GH_READ at 8+tRCDRD=10.
    REQUIRE_FALSE(controller.HasPendingPim(PimSource::INPUT));
}

TEST_CASE("Pending PIM work does not prevent another rank sleeping", "[pim][self-refresh]") {
    using namespace dramsim3;
    Config config("configs/HBM2_8Gb_x128_pim.ini", ".");
    config.enable_self_refresh = true;
    config.sref_threshold = 1;
    config.ranks = 2;
    Timing timing(config);
    Controller controller(0, config, timing);
    controller.EnqueueInputCommands({Command(CommandType::GH_READ,Address(0,1,0,0,0,0),0)},{10});
    controller.ClockTick();
    controller.ClockTick();
    REQUIRE(ControllerTestPeer::IsSelfRefreshing(controller,0));
    REQUIRE_FALSE(ControllerTestPeer::IsSelfRefreshing(controller,1));
}

TEST_CASE("GH issue slot is shared by weight and input queues", "[pim][gh-timing]") {
    using namespace dramsim3;
    Config config("configs/HBM2_8Gb_x128_pim.ini", ".");
    Timing timing(config);
    Controller controller(0, config, timing);
    ControllerTestPeer::PrepareOpenBanks(controller);
    CommandType read = CommandType::GH_READ;
    SECTION("ordinary read") {}
    SECTION("read with precharge") { read = CommandType::GH_READ_PRECHARGE; }
    controller.EnqueueWeightCommands({Command(read, Address(0, 0, 0, 0, 0, 0), 0)});
    controller.EnqueueInputCommands({Command(read, Address(0, 0, 2, 0, 0, 0), 0)}, {100});
    controller.ClockTick();
    REQUIRE(ControllerTestPeer::Weight(controller).empty());
    REQUIRE(ControllerTestPeer::Input(controller).size() == 1);
    controller.ClockTick();
    REQUIRE(ControllerTestPeer::Input(controller).empty());
}

TEST_CASE("LH reads retain parallel bank IO and two-cycle spacing", "[pim][gh-timing]") {
    using namespace dramsim3;
    Config config("configs/HBM2_8Gb_x128_pim.ini", ".");
    Timing timing(config);
    Controller controller(0, config, timing);
    ControllerTestPeer::PrepareOpenBanks(controller);
    std::vector<Command> commands;
    for (int bank=0; bank<16; ++bank) {
        int group = bank/4;
        int local_bank = bank%4;
        // The approved LH contract is all-bank, not a two-target broadcast.
        if (!(local_bank == 0 && (group == 0 || group == 2))) {
            ControllerTestPeer::OpenBank(controller, group, local_bank);
        }
        commands.emplace_back(CommandType::LH_READ,
                              Address(0, 0, group, local_bank, 0, 0), 0);
    }
    controller.EnqueueInputCommands(commands, std::vector<int>(16,100));
    controller.ClockTick();
    REQUIRE(ControllerTestPeer::Input(controller).empty());
    controller.EnqueueInputCommands(commands, std::vector<int>(16,101));
    controller.ClockTick();
    REQUIRE(ControllerTestPeer::Input(controller).size() == 16);
    controller.ClockTick();
    REQUIRE(ControllerTestPeer::Input(controller).empty());
}

TEST_CASE("PIM stage delays use the streaming dataflow clock", "[pim][gh-timing]") {
    using namespace dramsim3;
    Config config("configs/HBM2_8Gb_x128_pim.ini", ".");
    JedecDRAMSystem dramsys(config, ".", dummy_pim_callback,
                           dummy_pim_callback);
    uint64_t dataflow = 0x2c0e;
    int expected_out_count = 4;  // 1 * (3 + 16) - 14, then this tick's decrement
    int expected_in_count = 50;  // 1 * max(128 / 2, 16) - 14
    SECTION("GH streams at one-cycle array period") {}
    SECTION("LH streams at two-cycle array period") {
        dataflow = 0x2ece;
        expected_out_count = 23; // 2 * (3 + 16) - 14, then decrement
        expected_in_count = 18;  // 2 * max(128 / 16, 16) - 14
    }
    for (uint64_t address : {dataflow, uint64_t(0x200), uint64_t(0x402),
                             uint64_t(0x804), uint64_t(0x1)}) {
        REQUIRE(dramsys.AddTransaction(address)); // M=64, K=128, N=256
    }
    bool started_streaming = false;
    bool completed_tile = false;
    for (int cycle = 0; cycle < 2000; ++cycle) {
        dramsys.ClockTick();
        if (!started_streaming && dramsys.execution_.m_it == 1) {
            started_streaming = true;
            REQUIRE(dramsys.execution_.out_cnt == expected_out_count);
        }
        if (started_streaming && dramsys.execution_.iw_status == 3) {
            REQUIRE(dramsys.execution_.in_cnt == expected_in_count);
            completed_tile = true;
            break;
        }
    }
    REQUIRE(started_streaming);
    REQUIRE(completed_tile);
}

TEST_CASE("GH lookahead never bypasses actual issue readiness", "[pim][gh-timing]") {
    using namespace dramsim3;
    Config config("configs/HBM2_8Gb_x128_pim.ini", ".");
    Timing timing(config);
    Controller controller(0, config, timing);
    ControllerTestPeer::PrepareOpenBanks(controller);
    Command read(CommandType::GH_READ, Address(0, 0, 2, 0, 0, 0), 0);
    controller.EnqueueInputCommands({read}, {100});
    controller.ClockTick();
    REQUIRE(ControllerTestPeer::Input(controller).empty());
    REQUIRE_FALSE(controller.GetReadyCommand(read, 101).IsValid());
    REQUIRE(controller.GetReadyPimRead(read, 101, 1).cmd_type == CommandType::GH_READ);
    controller.EnqueueInputCommands({read}, {101});
    controller.ClockTick();
    REQUIRE(ControllerTestPeer::Input(controller).size() == 1);
    controller.ClockTick();
    REQUIRE(ControllerTestPeer::Input(controller).empty());
}

TEST_CASE("GH lookahead does not move activation into an earlier cycle", "[pim][gh-timing]") {
    using namespace dramsim3;
    Config config("configs/HBM2_8Gb_x128_pim.ini", ".");
    Timing timing(config);
    Controller controller(0, config, timing);
    ControllerTestPeer::PrepareOpenBanks(controller);
    Command read(CommandType::GH_READ, Address(0, 0, 0, 0, 0, 0), 0);
    Command readp(CommandType::GH_READ_PRECHARGE, read.addr, 0);
    controller.EnqueueWeightCommands({readp});
    controller.ClockTick();
    REQUIRE_FALSE(controller.GetReadyCommand(read, 101).IsValid());
    REQUIRE(controller.GetReadyCommand(read, 150).cmd_type == CommandType::PIM_ACTIVATE);
    REQUIRE_FALSE(controller.GetReadyPimRead(read, 101, 49).IsValid());
}

TEST_CASE("GH issue spacing follows tCCD_S", "[pim][gh-timing]") {
    using namespace dramsim3;
    Config config("configs/HBM2_8Gb_x128_pim.ini", ".");
    config.tCCD_S = 2;
    Timing timing(config);
    Controller controller(0, config, timing);
    ControllerTestPeer::PrepareOpenBanks(controller);
    controller.EnqueueInputCommands(
        {Command(CommandType::GH_READ, Address(0, 0, 0, 0, 0, 0), 0),
         Command(CommandType::GH_READ, Address(0, 0, 2, 0, 0, 0), 0)},
        {100, 100});
    controller.ClockTick();
    REQUIRE(ControllerTestPeer::Input(controller).size() == 1);
    controller.ClockTick();
    REQUIRE(ControllerTestPeer::Input(controller).size() == 1);
    controller.ClockTick();
    REQUIRE(ControllerTestPeer::Input(controller).empty());
}

TEST_CASE("Independent channels retain independent GH issue slots", "[pim][gh-timing]") {
    using namespace dramsim3;
    Config config("configs/HBM2_8Gb_x128_pim.ini", ".");
    Timing timing(config);
    Controller first(0, config, timing);
    Controller second(1, config, timing);
    for (Controller* controller : {&first, &second}) {
        ControllerTestPeer::PrepareOpenBanks(*controller);
        controller->EnqueueInputCommands(
            {Command(CommandType::GH_READ,
                     Address(controller->channel_id_, 0, 0, 0, 0, 0), 0)}, {100});
        controller->ClockTick();
        REQUIRE(ControllerTestPeer::Input(*controller).empty());
    }
}

TEST_CASE("PIM command batch preserves order and release pairing", "[pim]") {
    using namespace dramsim3;
    PimCommandBatch batch;
    Command first(CommandType::PIM_ACTIVATE, Address(0, 0, 0, 0, 1, 2), 3);
    Command second(CommandType::LH_READ, Address(1, 0, 0, 1, 4, 5), 6);
    batch.Add(PimSource::INPUT, PimOperation(first, 10));
    batch.Add(PimSource::INPUT, PimOperation(second, 11));
    REQUIRE(batch.operations.size() == 2);
    REQUIRE(batch.operations[0].second.commands[0].hex_addr == 3);
    REQUIRE(batch.operations[1].second.commands[0].hex_addr == 6);
    REQUIRE(batch.operations[0].second.release_cycle == 10);
    REQUIRE(batch.operations[1].second.release_cycle == 11);
}

TEST_CASE("PIM configuration resets execution progress", "[pim]") {
    using namespace dramsim3;
    Config config("configs/HBM2_8Gb_x128_pim.ini", ".");
    JedecDRAMSystem dramsys(config, ".", dummy_pim_callback,
                            dummy_pim_callback);
    dramsys.execution_.m = 17;
    dramsys.execution_.m_it = 12;
    dramsys.execution_.input_pending = true;
    dramsys.execution_.out_cnt = 20;
    REQUIRE(dramsys.AddTransaction(0x2c0e));
    dramsys.ClockTick();
    REQUIRE(dramsys.pim_trans_queue_.empty());
    REQUIRE(dramsys.execution_.m == 0);
    REQUIRE(dramsys.execution_.m_it == 0);
    REQUIRE_FALSE(dramsys.execution_.input_pending);
    REQUIRE(dramsys.execution_.out_cnt == -1);
}

TEST_CASE("PIM launch before dataflow configuration stays queued", "[pim]") {
    using namespace dramsim3;
    Config config("configs/HBM2_8Gb_x128_pim.ini", ".");
    JedecDRAMSystem dramsys(config, ".", dummy_pim_callback,
                            dummy_pim_callback);

    // Dimensions alone must not bypass the initialization guard.
    dramsys.execution_.m = 64;
    dramsys.execution_.k = 128;
    dramsys.execution_.n = 128;
    REQUIRE(dramsys.AddTransaction(0x1));
    dramsys.ClockTick();

    REQUIRE_FALSE(dramsys.execution_.in_pim);
    REQUIRE(dramsys.pim_trans_queue_.size() == 1);
    REQUIRE(dramsys.pim_trans_queue_.front().addr == 0x1);
}

TEST_CASE("PIM launch requires all three dimensions", "[pim]") {
    using namespace dramsim3;
    Config config("configs/HBM2_8Gb_x128_pim.ini", ".");
    JedecDRAMSystem dramsys(config, ".", dummy_pim_callback,
                            dummy_pim_callback);

    REQUIRE(dramsys.AddTransaction(0x2c0e));
    dramsys.ClockTick();
    dramsys.execution_.m = 64;
    dramsys.execution_.k = 128;
    dramsys.execution_.n = 128;
    bool ready = false;
    SECTION("missing M") { dramsys.execution_.m = 0; }
    SECTION("missing K") { dramsys.execution_.k = 0; }
    SECTION("missing N") { dramsys.execution_.n = 0; }
    SECTION("all configured") { ready = true; }
    REQUIRE(dramsys.AddTransaction(0x1));
    dramsys.ClockTick();
    REQUIRE(dramsys.execution_.in_pim == ready);
    REQUIRE(dramsys.pim_trans_queue_.empty() == ready);
}

TEST_CASE("PIM dataflow config preserves defaults and decoded values", "[pim]") {
    using namespace dramsim3;
    PimDataflowConfig config;
    REQUIRE_FALSE(config.configured);
    REQUIRE(config.mcf == 1);
    REQUIRE(config.ucf == 1);
    REQUIRE(config.mc == 1);
    REQUIRE(config.df == -1);
    REQUIRE(config.m_tile_size == 0);

    config.Load(PimTransactionDecoder::Decode(0x2ece));
    REQUIRE(config.configured);
    REQUIRE(config.mcf == 2);
    REQUIRE(config.ucf == 8);
    REQUIRE(config.mc == 16);
    REQUIRE(config.df == 1);
    REQUIRE(config.m_tile_size == 2048);
}

TEST_CASE("PIM transaction decoder covers compact field boundaries", "[pim]") {
    using namespace dramsim3;

    auto launch = PimTransactionDecoder::Decode(0x1);
    REQUIRE(launch.kind == PimTransactionKind::START_COMPUTATION);

    auto dataflow = PimTransactionDecoder::Decode(0x2c0e);
    REQUIRE(dataflow.kind == PimTransactionKind::LOAD_DATAFLOW_CONFIG);
    REQUIRE(dataflow.mcf == 2);
    REQUIRE(dataflow.ucf == 1);
    REQUIRE(dataflow.df == 0);
    REQUIRE(dataflow.m_tile_size == 2048);

    auto workload = PimTransactionDecoder::Decode(0x200);
    REQUIRE(workload.kind == PimTransactionKind::LOAD_WORKLOAD_CONFIG);
    REQUIRE(workload.load_type == 0);
    REQUIRE(workload.dim_value == 64);
    REQUIRE(workload.base_row == 0);

    uint64_t boundary_address =
        (0x3fffffULL << 35) |
        (0x7fffffffULL << 3) |
        (2ULL << 1);
    auto boundary = PimTransactionDecoder::Decode(boundary_address);
    REQUIRE(boundary.kind == PimTransactionKind::LOAD_WORKLOAD_CONFIG);
    REQUIRE(boundary.load_type == 2);
    REQUIRE(boundary.dim_value == 0x7fffffff);
    REQUIRE(boundary.base_row == 0x3fffff);

    uint64_t overlapping_marker = (3ULL << 1);
    REQUIRE(PimTransactionDecoder::Decode(overlapping_marker).kind ==
            PimTransactionKind::LOAD_DATAFLOW_CONFIG);
    auto extended = PimTransactionDecoder::Decode(
        (17ULL << 14) | (9ULL << 19) | 0x2c0e);
    REQUIRE(extended.kernel_size == 17);
    REQUIRE(extended.stride == 9);
    REQUIRE(extended.m_tile_size == 2048);
}

TEST_CASE("PIM transaction decoder reads compact single-array fields", "[pim]") {
    using namespace dramsim3;
    auto config = PimTransactionDecoder::Decode(0x2ece);
    REQUIRE(config.kind == PimTransactionKind::LOAD_DATAFLOW_CONFIG);
    REQUIRE(config.mcf == 2);
    REQUIRE(config.ucf == 8);
    REQUIRE(config.df == 1);
    REQUIRE(config.m_tile_size == 2048);

    auto workload = PimTransactionDecoder::Decode(0x64800000a02ULL);
    REQUIRE(workload.kind == PimTransactionKind::LOAD_WORKLOAD_CONFIG);
    REQUIRE(workload.load_type == 1);
    REQUIRE(workload.dim_value == 320);
    REQUIRE(workload.base_row == 201);
}
