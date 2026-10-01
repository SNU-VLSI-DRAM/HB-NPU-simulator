#include <cstdlib>

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
TEST_CASE("Split GEMM waits for resident work on channel three through refresh",
          "[instruction-pipeline]") {
    char temp[] = "/tmp/hbnpu-resident-pipeline-XXXXXX";
    REQUIRE(mkdtemp(temp) != nullptr);
    Config hw("configs/HBM2_8Gb_x128_pim.ini", temp);
    hw.tile_instructions = true;
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
    REQUIRE_FALSE(hw.tile_instructions);
    REQUIRE(hw.instruction_capacity == 16);
    hw.tile_instructions = true;
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
    REQUIRE(system.pim_trans_queue_.empty());
}
