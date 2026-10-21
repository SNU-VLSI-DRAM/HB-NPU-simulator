#include <cstdlib>
#include <fstream>
#include <sstream>
#include <algorithm>

#include "catch.hpp"
#include "dram_system.h"
using namespace dramsim3;
namespace {
class PipelineSystem : public JedecDRAMSystem {
   public:
    using JedecDRAMSystem::JedecDRAMSystem;
    void DelayChannelThree() {
        Command command(CommandType::GH_READ, Address(3, 0, 2, 1, 2133, 0), 0);
        ctrls_[3]->EnqueuePimOperation(PimSource::INPUT,
                                       PimOperation(command, 3000));
    }
    void DelayResidentChannelThree() {
        Command command(CommandType::GH_READ, Address(3, 0, 0, 0, 0, 0), 0);
        ctrls_[3]->EnqueuePimOperation(PimSource::WEIGHT,
                                       PimOperation(command, 3000));
    }
    bool RefreshActive() const {
        for (auto* controller : ctrls_)
            if (controller->IsInRef()) return true;
        return false;
    }
};
}  // namespace
// Catches treating command issue as operand arrival, applying the delay to LH,
// or waiting read_delay between each successive GH command.
TEST_CASE("GH operands arrive after read latency without slowing issue cadence",
          "[instruction-pipeline][gh-arrival]") {
    bool local = false, lockstep = true, delayed_channel = false;
    uint64_t m = 64;
    SECTION("GH row") {}
    SECTION("LH row") { local = true; }
    SECTION("GH vector") { lockstep = false; }
    SECTION("LH vector") { local = true; lockstep = false; }
    SECTION("GH single vector tail") { m = 1; }
    SECTION("GH partial row tail") { m = 33; }
    SECTION("slow resident channel") { delayed_channel = true; }
    char temp[] = "/tmp/hbnpu-gh-arrival-XXXXXX";
    REQUIRE(mkdtemp(temp) != nullptr);
    const std::string config_path = std::string(temp) + "/timing.ini";
    {
        std::ifstream original("configs/HBM2_8Gb_x128_pim.ini");
        std::ofstream config(config_path);
        config << original.rdbuf() << "\n[hbnpu]\nexecution_mode = tile_instructions\n"
               << "gh_read_delay = true\nrow_lockstep = " << (lockstep ? "true" : "false") << "\n";
    }
    Config hw(config_path, temp);
    // Deliberately larger than a row: catches draining before the last return.
    hw.read_delay = 80;
    auto callback = [](uint64_t) {};
    PipelineSystem system(hw, temp, callback, callback);
    const uint64_t words[] = {0xb711070000000000ULL,
                              0x8000000000ULL | m, 0x8000000080ULL,
                              2133, 0, 3000,
                              local ? 0x80201ULL : 0x10200ULL,
                              0x0020001000000008ULL,
                              0xb712010000000001ULL, 0,
                              0xb715020000000002ULL, 0, 0,
                              0xb716020000000003ULL, 0, 0,
                              0xb714000000000004ULL};
    for (auto word : words) REQUIRE(system.AddTransaction(word));
    int stream_request = -1, output_timer = -1, stream_consumed = -1;
    bool injected = false;
    for (int tick = 0; tick < 10000 && !system.turn_off; ++tick) {
        system.ClockTick();
        if (delayed_channel && !injected && system.execution_.weight_pending) {
            system.DelayResidentChannelThree();
            injected = true;
        }
        if (stream_request < 0 && system.execution_.input_pending) stream_request = tick;
        if (output_timer < 0 && system.execution_.out_cnt >= 0) output_timer = tick;
        if (stream_consumed < 0 && system.execution_.iw_status == 3) stream_consumed = tick;
    }
    REQUIRE(system.turn_off);
    REQUIRE(stream_request >= 0);
    REQUIRE(output_timer >= 0);
    REQUIRE(stream_consumed >= 0);
#ifdef CMD_TRACE
    for (int ch = 0; ch < 8; ++ch) {
        std::ifstream trace(std::string(temp) + "/dramsim3ch_" + std::to_string(ch) + "cmd.trace");
        std::vector<int> resident, input;
        std::string line;
        while (std::getline(trace, line)) {
            std::istringstream fields(line);
            int cycle, channel, rank, bg, bank;
            unsigned row, column;
            std::string kind;
            fields >> cycle >> kind >> channel >> rank >> bg >> bank >> std::hex >> row >> column;
            if (kind.find("gh_read") != 0 && kind.find("lh_read") != 0) continue;
            if (row < 100) resident.push_back(cycle);
            if (row >= 2133 && row < 2200) input.push_back(cycle);
        }
        REQUIRE_FALSE(resident.empty());
        REQUIRE_FALSE(input.empty());
        // Resident reads use GH in both dataflows. Readiness gates streaming.
        REQUIRE(stream_request >= resident.back() + 80);
        // Paired GH bank beats are one cycle apart. LH is one all-bank beat.
        const int first_vector_issue = input.at(local ? 0 : 1);
        REQUIRE(output_timer == first_vector_issue + (local ? 1 : 80));
        REQUIRE(stream_consumed == input.back() + (local ? 1 : 80));
        if (!local) {
            REQUIRE(input.at(1) - input.at(0) == 1);
            if (m > 1) {
                REQUIRE(input.at(3) - input.at(2) == 1);
                REQUIRE(input.at(2) - input.at(0) == 2);
            }
        }
    }
#endif
}

TEST_CASE("Split GEMM waits for resident work on channel three through refresh",
          "[instruction-pipeline]") {
    char temp[] = "/tmp/hbnpu-resident-pipeline-XXXXXX";
    REQUIRE(mkdtemp(temp) != nullptr);
    Config hw("configs/HBM2_8Gb_x128_pim.ini", temp);
    hw.instruction_capacity = 1;
    hw.tREFI = 1000;
    auto callback = [](uint64_t) {};
    PipelineSystem system(hw, temp, callback, callback);
    const uint64_t words[] = {0xb711070000000000ULL,
                              0x8000000040ULL,
                              0x4000000080ULL,
                              2133,
                              0,
                              2133,
                              0x10200,
                              0x0020001000000008ULL,
                              0xb712010000000001ULL,
                              0,
                              0xb715020000000002ULL,
                              0,
                              0,
                              0xb716020000000003ULL,
                              0,
                              0,
                              0xb714000000000004ULL};
    for (auto word : words) REQUIRE(system.AddTransaction(word));
    bool injected = false, saw_refresh = false;
    for (int tick = 0; tick < 2000; ++tick) {
        system.ClockTick();
        saw_refresh |= system.RefreshActive();
        if (!injected && system.execution_.weight_pending) {
            system.DelayResidentChannelThree();
            injected = true;
        }
        REQUIRE_FALSE(system.execution_.input_pending);
        REQUIRE(system.execution_.m_it == 0);
    }
    REQUIRE(injected);
    REQUIRE(saw_refresh);
    REQUIRE_FALSE(system.turn_off);
    for (int tick = 2000; tick < 15000 && !system.turn_off; ++tick)
        system.ClockTick();
    REQUIRE(system.turn_off);
}

TEST_CASE("Instruction drain waits for input pending only in channel three",
          "[instruction-pipeline]") {
    char temp[] = "/tmp/hbnpu-pipeline-XXXXXX";
    REQUIRE(mkdtemp(temp) != nullptr);
    Config hw("configs/HBM2_8Gb_x128_pim.ini", temp);
    REQUIRE(hw.instruction_capacity == 16);
    hw.instruction_capacity = 1;
    bool require_refresh = false;
    SECTION("ordinary delayed input") {}
    SECTION("delayed input survives refresh") {
        hw.tREFI = 1000;
        require_refresh = true;
    }
    auto callback = [](uint64_t) {};
    PipelineSystem system(hw, temp, callback, callback);
    const uint64_t words[] = {0xb711070000000000ULL,
                              0x8000000040ULL,
                              0x4000000080ULL,
                              2133,
                              0,
                              2133,
                              0x10200,
                              0x0020001000000008ULL,
                              0xb712010000000001ULL,
                              0,
                              0xb713020000000002ULL,
                              0,
                              0,
                              0xb714000000000003ULL};
    for (auto word : words) REQUIRE(system.AddTransaction(word));
    bool injected = false;
    bool saw_refresh = false;
    for (int tick = 0; tick < 2000; ++tick) {
        system.ClockTick();
        saw_refresh |= system.RefreshActive();
        if (!injected && system.execution_.input_pending) {
            system.DelayChannelThree();
            injected = true;
        }
    }
    REQUIRE(injected);
    if (require_refresh) REQUIRE(saw_refresh);
    REQUIRE_FALSE(system.turn_off);
    for (int tick = 2000; tick < 10000 && !system.turn_off; ++tick)
        system.ClockTick();
    REQUIRE(system.turn_off);
}
