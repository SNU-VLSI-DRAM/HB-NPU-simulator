#include "catch.hpp"
#include "pim_operation.h"
#include "channel_state.h"

using namespace dramsim3;
namespace {
PimOperation Gang(int first, int rank = 0, CommandType type = CommandType::PIM_ACTIVATE) {
    std::vector<Command> commands;
    for (int bank = first; bank < first + 4; ++bank)
        commands.emplace_back(type, Address(0, rank, bank / 4, bank % 4, 7, 0), 0);
    return PimOperation(type == CommandType::PIM_ACTIVATE ? PimOpcode::GANG_ACT : PimOpcode::GANG_PRE, commands);
}
}

TEST_CASE("PIM gangs reserve four activation slots atomically", "[pim][gang]") {
    Config config("configs/HBM2_8Gb_x128_pim.ini", ".");
    config.tFAW = 30;
    config.ranks = 2;
    Timing timing(config);
    ChannelState state(config, timing);
    REQUIRE(state.GetReadyPimOperation(Gang(0), 0).IsValid());
    state.IssuePimOperation(Gang(0), 0);
    for (int bank = 0; bank < 4; ++bank) REQUIRE(state.OpenRow(0, 0, bank) == 7);
    REQUIRE_FALSE(state.GetReadyPimOperation(Gang(4), 29).IsValid());
    REQUIRE(state.GetReadyPimOperation(Gang(4, 1), 1).IsValid());
    REQUIRE(state.GetReadyPimOperation(Gang(4), 30).IsValid());
    state.IssuePimOperation(Gang(4), 30);
    REQUIRE_FALSE(state.GetReadyPimOperation(Gang(8), 31).IsValid());
    REQUIRE(state.GetReadyPimOperation(Gang(8), 60).IsValid());
}

TEST_CASE("One recent activation blocks a gang without partial effects", "[pim][gang]") {
    Config config("configs/HBM2_8Gb_x128_pim.ini", ".");
    config.tFAW = 30;
    Timing timing(config);
    ChannelState state(config, timing);
    CommandType act = CommandType::ACTIVATE;
    SECTION("normal activation") {}
    SECTION("PIM activation") { act = CommandType::PIM_ACTIVATE; }
    state.UpdateTimingAndStates(Command(act, Address(0,0,3,3,1,0),0),0);
    REQUIRE_FALSE(state.GetReadyPimOperation(Gang(0),1).IsValid());
    REQUIRE_FALSE(state.IssuePimOperation(Gang(0),1));
    for (int bank=0; bank<4; ++bank) REQUIRE_FALSE(state.IsRowOpen(0,0,bank));
    REQUIRE(state.IssuePimOperation(Gang(0),30));
}

TEST_CASE("Gang validates targets and waits for every precharge recovery", "[pim][gang]") {
    Config config("configs/HBM2_8Gb_x128_pim.ini", ".");
    Timing timing(config);
    ChannelState state(config, timing);
    auto invalid = Gang(0);
    SECTION("duplicate") { invalid.commands[3] = invalid.commands[0]; }
    SECTION("wrong width") { invalid.commands.pop_back(); }
    SECTION("cross rank") { invalid.commands[3].addr.rank = 1; }
    SECTION("cross channel") { invalid.commands[3].addr.channel = 1; }
    SECTION("different row") { invalid.commands[3].addr.row = 9; }
    REQUIRE_THROWS_AS(state.GetReadyPimOperation(invalid,0), std::invalid_argument);
    REQUIRE(state.IssuePimOperation(Gang(0),0));
    state.UpdateTimingAndStates(Command(CommandType::PIM_WRITE,Address(0,0,0,3,7,0),0),100);
    REQUIRE_FALSE(state.IssuePimOperation(Gang(0,0,CommandType::PRECHARGE),101));
    for (int bank=0; bank<4; ++bank) REQUIRE(state.IsRowOpen(0,0,bank));
    REQUIRE(state.IssuePimOperation(Gang(0,0,CommandType::PRECHARGE),200));
    REQUIRE_FALSE(state.IssuePimOperation(Gang(0),200 + config.tRP - 1));
    REQUIRE(state.IssuePimOperation(Gang(0),200 + config.tRP));
}

TEST_CASE("LH data intent prepares sixteen banks using four gangs", "[pim][gang]") {
    Config config("configs/HBM2_8Gb_x128_pim.ini", ".");
    config.tFAW = 30;
    Timing timing(config);
    ChannelState state(config,timing);
    std::vector<Command> reads;
    for (int bank=0; bank<16; ++bank)
        reads.emplace_back(CommandType::LH_READ,Address(0,0,bank/4,bank%4,7,0),0);
    PimOperation read(PimOpcode::LH_READ,reads);
    for (int cycle : {0,30,60,90}) {
        auto ready=state.GetReadyPimOperation(read,cycle);
        REQUIRE(ready.opcode == PimOpcode::GANG_ACT);
        REQUIRE(ready.commands.front().Bankgroup() == cycle/30);
        REQUIRE(state.IssuePimOperation(ready,cycle));
        REQUIRE_FALSE(state.IssuePimOperation(read,cycle+1));
    }
    REQUIRE(state.IssuePimOperation(read,90+config.tRCDRD));
    for (int bank=0; bank<16; ++bank) REQUIRE(state.RowHitCount(0,bank/4,bank%4)==1);
}

TEST_CASE("LH preparation skips open banks and uses a single-bank remainder", "[pim][gang]") {
    Config config("configs/HBM2_8Gb_x128_pim.ini", ".");
    config.tFAW = 30;
    config.tRRD_L = 1000;
    config.tRRD_S = 1000;
    Timing timing(config);
    ChannelState state(config,timing);
    std::vector<Command> commands;
    CommandType type = CommandType::LH_READ;
    PimOpcode opcode = PimOpcode::LH_READ;
    SECTION("LH read") {}
    SECTION("LH write") { type=CommandType::PIM_WRITE; opcode=PimOpcode::LH_WRITE; }
    SECTION("LH read with precharge") { type=CommandType::LH_READ_PRECHARGE; opcode=PimOpcode::LH_READ_PRECHARGE; }
    SECTION("LH write with precharge") { type=CommandType::PIM_WRITE_PRECHARGE; opcode=PimOpcode::LH_WRITE_PRECHARGE; }
    for (int bank=0; bank<16; ++bank) {
        Address address(0,0,bank/4,bank%4,7,0);
        commands.emplace_back(type,address,0);
        if (bank<3) state.UpdateTimingAndStates(Command(CommandType::PIM_ACTIVATE,address,0),0);
    }
    PimOperation operation(opcode,commands);
    for (int cycle : {30,60,90}) {
        auto ready=state.GetReadyPimOperation(operation,cycle);
        REQUIRE(ready.opcode==PimOpcode::GANG_ACT);
        REQUIRE(ready.commands.front().Bankgroup()*4+ready.commands.front().Bank()==3+4*(cycle/30-1));
        REQUIRE(state.IssuePimOperation(ready,cycle));
    }
    auto last=state.GetReadyPimOperation(operation,120);
    REQUIRE(last.opcode==PimOpcode::PIM_ACT);
    REQUIRE(last.commands.front().Bankgroup()==3);
    REQUIRE(last.commands.front().Bank()==3);
    REQUIRE(state.IssuePimOperation(last,120));
    REQUIRE_FALSE(state.IssuePimOperation(operation,121));
    for (int bank=0; bank<16; ++bank) REQUIRE(state.RowHitCount(0,bank/4,bank%4)==0);
    REQUIRE(state.IssuePimOperation(operation,150));
}

TEST_CASE("GH preparation changes only its addressed bank", "[pim][gang]") {
    Config config("configs/HBM2_8Gb_x128_pim.ini", ".");
    Timing timing(config);
    ChannelState state(config,timing);
    PimOperation write(Command(CommandType::PIM_WRITE,Address(0,0,2,3,5,0),0));
    auto activate=state.GetReadyPimOperation(write,0);
    REQUIRE(activate.opcode==PimOpcode::PIM_ACT);
    REQUIRE(state.IssuePimOperation(activate,0));
    for (int bank=0; bank<16; ++bank) REQUIRE(state.IsRowOpen(0,bank/4,bank%4)==(bank==11));
    write.commands.front().addr.row=6;
    auto pre=state.GetReadyPimOperation(write,100);
    REQUIRE(pre.opcode==PimOpcode::PIM_PRE);
    REQUIRE(state.IssuePimOperation(pre,100));
    REQUIRE_FALSE(state.GetReadyPimOperation(write,100+config.tRP-1).IsValid());
    REQUIRE(state.GetReadyPimOperation(write,100+config.tRP).opcode==PimOpcode::PIM_ACT);
}

TEST_CASE("LH rejects partial and incompatible broadcast targets", "[pim][gang]") {
    Config config("configs/HBM2_8Gb_x128_pim.ini", ".");
    Timing timing(config);
    ChannelState state(config,timing);
    std::vector<Command> commands;
    for (int bank=0; bank<16; ++bank) commands.emplace_back(CommandType::LH_READ,Address(0,0,bank/4,bank%4,0,0),0);
    SECTION("partial") { commands.pop_back(); }
    SECTION("column mismatch") { commands[15].addr.column=1; }
    SECTION("opcode mismatch") { commands[15].cmd_type=CommandType::GH_READ; }
    SECTION("out of range bank") { commands[15].addr.bank=4; }
    REQUIRE_THROWS_AS(state.GetReadyPimOperation(PimOperation(PimOpcode::LH_READ,commands),0),std::invalid_argument);
}
