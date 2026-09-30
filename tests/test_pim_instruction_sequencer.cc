#include "catch.hpp"
#include "pim_instruction_sequencer.h"
#include <vector>
using namespace dramsim3;
namespace {
std::vector<uint64_t> Program(bool end=true, bool multi=false) {
    std::vector<uint64_t> words={0xb711070000000000ULL,
        0x8000000040ULL | (multi ? 128 : 64), 0x4000000100ULL,
        2133,0,2133,0x10200,0x0020001000000008ULL};
    // M=64 or 192: use explicit 64-sized groups and two K blocks/group.
    uint64_t seq=1;
    for (uint64_t m : (multi ? std::vector<uint64_t>{0,64,128} : std::vector<uint64_t>{0})) {
        words.push_back(0xb712010000000000ULL|seq++); words.push_back(m);
        words.push_back(0xb713020000000000ULL|seq++); words.push_back(m); words.push_back(0);
        words.push_back(0xb713020000000000ULL|seq++); words.push_back(m); words.push_back(128);
    }
    if (end) words.push_back(0xb714000000000000ULL|seq);
    return words;
}
void Run(const std::vector<uint64_t>& words,PimInstructionBuffer& buffer,
         PimInstructionSequencer& sequencer,int period=1,int ticks=15000) {
    size_t next=0;
    for (int tick=0;tick<ticks && !sequencer.Complete();++tick) {
        buffer.Tick(tick); sequencer.Tick(tick,buffer,PimTickFeedback{});
        if (next<words.size() && tick%period==0 && buffer.PushWord(words[next])) ++next;
    }
}
}
TEST_CASE("STORE dispatch does not block its required compute", "[instruction-sequencer]") {
    Config hw("configs/HBM2_8Gb_x128_pim.ini","/tmp");
    PimInstructionBuffer buffer(1); PimInstructionSequencer sequencer(hw);
    auto words=Program();
    for (int tick=0;tick<16;++tick) {
        buffer.Tick(tick); sequencer.Tick(tick,buffer,PimTickFeedback{});
        if (tick<13) buffer.PushWord(words[tick]);
    }
    // Capacity one inserts a header wait before each previously occupied slot.
    // Final TILE payload completes at 15, eligible on 16.
    buffer.Tick(16); sequencer.Tick(16,buffer,PimTickFeedback{});
    REQUIRE(sequencer.Stats().dispatched_instructions==3);
    REQUIRE(sequencer.Stats().retired_instructions==1);
    REQUIRE_FALSE(sequencer.Complete());
    REQUIRE(sequencer.Snapshot().output_valid==0);
}
TEST_CASE("Capacity one and sparse uploads complete all explicit groups", "[instruction-sequencer]") {
    Config hw("configs/HBM2_8Gb_x128_pim.ini","/tmp");
    for (int period : {1,100}) {
        PimInstructionBuffer buffer(1); PimInstructionSequencer sequencer(hw);
        Run(Program(true,true),buffer,sequencer,period);
        REQUIRE(sequencer.Complete()); REQUIRE(buffer.Empty());
        REQUIRE(sequencer.Stats().dispatched_instructions==11);
        REQUIRE(sequencer.Stats().retired_instructions==11);
        REQUIRE(sequencer.Stats().end_cycle>=sequencer.Stats().backend_drain_cycle);
        if (period==100) REQUIRE(sequencer.Stats().instruction_starvation_cycles>0);
    }
}
TEST_CASE("Backend drain without END cannot complete", "[instruction-sequencer]") {
    Config hw("configs/HBM2_8Gb_x128_pim.ini","/tmp");
    PimInstructionBuffer buffer(16); PimInstructionSequencer sequencer(hw);
    Run(Program(false),buffer,sequencer);
    REQUIRE(sequencer.Stats().backend_drained); REQUIRE_FALSE(sequencer.Complete());
    REQUIRE(sequencer.Stats().retired_instructions==4);
}
TEST_CASE("Illegal programs fail before resource stalls hide them", "[instruction-sequencer]") {
    Config hw("configs/HBM2_8Gb_x128_pim.ini","/tmp");
    auto words=Program();
    SECTION("missing CONFIG") { words={0xb714000000000000ULL}; }
    SECTION("early END") { words.resize(8); words.push_back(0xb714000000000001ULL); }
    SECTION("wrong opcode where STORE required") {
        words[8]=0xb713020000000001ULL; words.insert(words.begin()+10,0);
    }
    SECTION("wrong compute coordinates while compute busy") { words[14]=64; }
    SECTION("CONFIG replacement") { words[13]=0xb711070000000003ULL; words.resize(21,0); }
    SECTION("extra words buffered behind END") { words.push_back(0xb714000000000005ULL); }
    PimInstructionBuffer buffer(16); PimInstructionSequencer sequencer(hw);
    REQUIRE_THROWS_AS(Run(words,buffer,sequencer),std::invalid_argument);
}
TEST_CASE("Refresh freezes execution but not free descriptor dispatch", "[instruction-sequencer]") {
    Config hw("configs/HBM2_8Gb_x128_pim.ini","/tmp");
    PimInstructionBuffer buffer(16); PimInstructionSequencer sequencer(hw);
    auto words=Program(); for (auto word:words) REQUIRE(buffer.PushWord(word));
    PimTickFeedback feedback; feedback.in_refresh=true;
    for (int tick=0;tick<30;++tick) {
        buffer.Tick(tick); auto result=sequencer.Tick(tick,buffer,feedback);
        REQUIRE(result.events.empty());
    }
    REQUIRE(sequencer.Stats().dispatched_instructions==3);
    REQUIRE(sequencer.Stats().retired_instructions==1);
    REQUIRE_FALSE(sequencer.Stats().backend_drained);
}
