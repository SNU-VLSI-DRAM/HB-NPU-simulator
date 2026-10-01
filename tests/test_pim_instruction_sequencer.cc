#include <vector>

#include "catch.hpp"
#include "pim_instruction_sequencer.h"
using namespace dramsim3;
namespace {
std::vector<uint64_t> Program(bool end = true, bool multi = false) {
    std::vector<uint64_t> words = {0xb711070000000000ULL,
                                   0x8000000040ULL | (multi ? 128 : 64),
                                   0x4000000100ULL,
                                   2133,
                                   0,
                                   2133,
                                   0x10200,
                                   0x0020001000000008ULL};
    // M=64 or 192: use explicit 64-sized groups and two K blocks/group.
    uint64_t seq = 1;
    for (uint64_t m : (multi ? std::vector<uint64_t>{0, 64, 128}
                             : std::vector<uint64_t>{0})) {
        words.push_back(0xb712010000000000ULL | seq++);
        words.push_back(m);
        words.push_back(0xb713020000000000ULL | seq++);
        words.push_back(m);
        words.push_back(0);
        words.push_back(0xb713020000000000ULL | seq++);
        words.push_back(m);
        words.push_back(128);
    }
    if (end) words.push_back(0xb714000000000000ULL | seq);
    return words;
}
void Run(const std::vector<uint64_t>& words, PimInstructionBuffer& buffer,
         PimInstructionSequencer& sequencer, int period = 1,
         int ticks = 15000) {
    size_t next = 0;
    for (int tick = 0; tick < ticks && !sequencer.Complete(); ++tick) {
        buffer.Tick(tick);
        sequencer.Tick(tick, buffer, PimTickFeedback{});
        if (next < words.size() && tick % period == 0 &&
            buffer.PushWord(words[next]))
            ++next;
    }
}
std::vector<uint64_t> SplitProgram(bool multi = false) {
    auto fused = Program(true, multi);
    std::vector<uint64_t> split;
    uint32_t sequence = 0;
    for (size_t i = 0; i < fused.size();) {
        auto header = PimInstructionCodec::DecodeHeader(fused[i]);
        const size_t end = i + 1 + header.payload_words;
        for (int op :
             (header.opcode == PimInstructionOpcode::TILE_BLOCK
                  ? std::vector<int>{5, 6}
                  : std::vector<int>{static_cast<int>(header.opcode)})) {
            auto h = header;
            h.opcode = static_cast<PimInstructionOpcode>(op);
            h.sequence = sequence++;
            split.push_back(PimInstructionCodec::EncodeHeader(h));
            split.insert(split.end(), fused.begin() + i + 1,
                         fused.begin() + end);
        }
        i = end;
    }
    return split;
}
}  // namespace
TEST_CASE("Resident load alone never starts streamed GEMM",
          "[instruction-sequencer]") {
    Config hw("configs/HBM2_8Gb_x128_pim.ini", "/tmp");
    PimInstructionBuffer buffer(1);
    PimInstructionSequencer sequencer(hw);
    auto words = Program();
    words.resize(13);
    words[10] = 0xb715020000000002ULL;
    size_t next = 0;
    bool resident_issued = false;
    for (int tick = 0; tick < 1000; ++tick) {
        buffer.Tick(tick);
        auto result = sequencer.Tick(tick, buffer, PimTickFeedback{});
        for (const auto& op : result.batch.operations) {
            REQUIRE(op.first == PimSource::WEIGHT);
            resident_issued = true;
        }
        if (next < words.size() && buffer.PushWord(words[next])) ++next;
    }
    REQUIRE(resident_issued);
    REQUIRE(sequencer.Stats().dispatched_instructions == 3);
    REQUIRE(sequencer.Stats().retired_instructions == 2);
    REQUIRE_FALSE(sequencer.Complete());
    REQUIRE(sequencer.Snapshot().m_it == 0);
    REQUIRE(sequencer.Snapshot().output_valid == 0);
    REQUIRE(sequencer.Stats().instruction_starvation_cycles > 0);
    // Upload GEMM only after the resident load has long since completed.
    auto split = SplitProgram();
    next = 13;
    for (int tick = 1000; tick < 15000 && !sequencer.Complete(); ++tick) {
        buffer.Tick(tick);
        sequencer.Tick(tick, buffer, PimTickFeedback{});
        if (next < split.size() && buffer.PushWord(split[next])) ++next;
    }
    REQUIRE(sequencer.Complete());
    REQUIRE(sequencer.Stats().retired_instructions == 7);
}

TEST_CASE("Split pairs preserve group order under sparse upload and refresh",
          "[instruction-sequencer]") {
    Config hw("configs/HBM2_8Gb_x128_pim.ini", "/tmp");
    for (int period : {1, 100}) {
        PimInstructionBuffer buffer(1);
        PimInstructionSequencer sequencer(hw);
        auto words = SplitProgram(true);
        size_t next = 0;
        for (int tick = 0; tick < 30000 && !sequencer.Complete(); ++tick) {
            PimTickFeedback feedback;
            feedback.in_refresh = tick % 1000 < 100;
            buffer.Tick(tick);
            auto result = sequencer.Tick(tick, buffer, feedback);
            if (feedback.in_refresh) REQUIRE(result.batch.operations.empty());
            if (next < words.size() && tick % period == 0 &&
                buffer.PushWord(words[next]))
                ++next;
        }
        REQUIRE(sequencer.Complete());
        REQUIRE(buffer.Empty());
        REQUIRE(sequencer.Stats().retired_instructions == 17);
        REQUIRE(sequencer.Stats().dispatched_instructions == 17);
    }
}

TEST_CASE("Split pair violations fail even while the resident load is pending",
          "[instruction-sequencer]") {
    Config hw("configs/HBM2_8Gb_x128_pim.ini", "/tmp");
    auto words = SplitProgram();
    SECTION("GEMM without load") { words[10] = 0xb716020000000002ULL; }
    SECTION("duplicate load") { words[13] = 0xb715020000000003ULL; }
    SECTION("different K tile") { words[15] = 128; }
    SECTION("fused tile cannot consume explicit load") {
        words[13] = 0xb713020000000003ULL;
    }
    SECTION("END cannot replace GEMM") {
        words.resize(14);
        words[13] = 0xb714000000000003ULL;
    }
    PimInstructionBuffer buffer(16);
    PimInstructionSequencer sequencer(hw);
    for (auto word : words) REQUIRE(buffer.PushWord(word));
    PimTickFeedback feedback;
    feedback.weight_pending = true;
    REQUIRE_THROWS_AS(([&]() {
                          for (int tick = 0; tick < 100; ++tick) {
                              buffer.Tick(tick);
                              sequencer.Tick(tick, buffer, feedback);
                          }
                      })(),
                      std::invalid_argument);
}

TEST_CASE("Split GEMM dispatches during resident load but waits for readiness",
          "[instruction-sequencer]") {
    Config hw("configs/HBM2_8Gb_x128_pim.ini", "/tmp");
    PimInstructionBuffer buffer(1);
    PimInstructionSequencer sequencer(hw);
    auto words = Program();
    words.resize(10);
    const std::vector<uint64_t> tail = {
        0xb715020000000002ULL, 0, 0,   0xb716020000000003ULL, 0, 0,
        0xb715020000000004ULL, 0, 128, 0xb716020000000005ULL, 0, 128,
        0xb714000000000006ULL};
    words.insert(words.end(), tail.begin(), tail.end());
    size_t next = 0;
    bool streamed = false;
    for (int tick = 0; tick < 15000 && !sequencer.Complete(); ++tick) {
        PimTickFeedback feedback;
        feedback.weight_pending = tick < 500;
        buffer.Tick(tick);
        auto result = sequencer.Tick(tick, buffer, feedback);
        for (const auto& op : result.batch.operations) {
            if (op.first == PimSource::INPUT) {
                REQUIRE(tick >= 500);
                streamed = true;
            }
        }
        if (tick == 400) {
            REQUIRE(sequencer.Stats().dispatched_instructions == 4);
            REQUIRE(sequencer.Stats().retired_instructions == 1);
        }
        if (next < words.size() && buffer.PushWord(words[next])) ++next;
    }
    REQUIRE(streamed);
    REQUIRE(sequencer.Complete());
    REQUIRE(sequencer.Stats().retired_instructions == 7);
}

TEST_CASE("Fused tiles can replace alternating complete load GEMM pairs",
          "[instruction-sequencer]") {
    Config hw("configs/HBM2_8Gb_x128_pim.ini", "/tmp");
    const auto split = SplitProgram(true);
    std::vector<uint64_t> mixed;
    uint32_t sequence = 0;
    int pair = 0;
    for (size_t i = 0; i < split.size();) {
        auto header = PimInstructionCodec::DecodeHeader(split[i]);
        size_t next = i + 1 + header.payload_words;
        if (header.opcode == PimInstructionOpcode::LOAD_INPUTS &&
            pair++ % 2 == 0) {
            header.opcode = PimInstructionOpcode::TILE_BLOCK;
            next += 3;  // Replace the entire LOAD/GEMM pair, never half a pair.
        }
        header.sequence = sequence++;
        mixed.push_back(PimInstructionCodec::EncodeHeader(header));
        mixed.insert(mixed.end(), split.begin() + i + 1,
                     split.begin() + i + 1 + header.payload_words);
        i = next;
    }
    for (int period : {1, 100}) {
        PimInstructionBuffer buffer(1);
        PimInstructionSequencer sequencer(hw);
        Run(mixed, buffer, sequencer, period, 30000);
        REQUIRE(sequencer.Complete());
        REQUIRE(buffer.Empty());
        REQUIRE(sequencer.Stats().retired_instructions == 14);
        REQUIRE(sequencer.Stats().dispatched_instructions == 14);
    }
}

TEST_CASE("STORE dispatch does not block its required compute",
          "[instruction-sequencer]") {
    Config hw("configs/HBM2_8Gb_x128_pim.ini", "/tmp");
    PimInstructionBuffer buffer(1);
    PimInstructionSequencer sequencer(hw);
    auto words = Program();
    for (int tick = 0; tick < 16; ++tick) {
        buffer.Tick(tick);
        sequencer.Tick(tick, buffer, PimTickFeedback{});
        if (tick < 13) buffer.PushWord(words[tick]);
    }
    // Capacity one inserts a header wait before each previously occupied slot.
    // Final TILE payload completes at 15, eligible on 16.
    buffer.Tick(16);
    sequencer.Tick(16, buffer, PimTickFeedback{});
    REQUIRE(sequencer.Stats().dispatched_instructions == 3);
    REQUIRE(sequencer.Stats().retired_instructions == 1);
    REQUIRE_FALSE(sequencer.Complete());
    REQUIRE(sequencer.Snapshot().output_valid == 0);
}
TEST_CASE("Capacity one and sparse uploads complete all explicit groups",
          "[instruction-sequencer]") {
    Config hw("configs/HBM2_8Gb_x128_pim.ini", "/tmp");
    for (int period : {1, 100}) {
        PimInstructionBuffer buffer(1);
        PimInstructionSequencer sequencer(hw);
        Run(Program(true, true), buffer, sequencer, period);
        REQUIRE(sequencer.Complete());
        REQUIRE(buffer.Empty());
        REQUIRE(sequencer.Stats().dispatched_instructions == 11);
        REQUIRE(sequencer.Stats().retired_instructions == 11);
        REQUIRE(sequencer.Stats().end_cycle >=
                sequencer.Stats().backend_drain_cycle);
        if (period == 100)
            REQUIRE(sequencer.Stats().instruction_starvation_cycles > 0);
    }
}
TEST_CASE("Backend drain without END cannot complete",
          "[instruction-sequencer]") {
    Config hw("configs/HBM2_8Gb_x128_pim.ini", "/tmp");
    PimInstructionBuffer buffer(16);
    PimInstructionSequencer sequencer(hw);
    Run(Program(false), buffer, sequencer);
    REQUIRE(sequencer.Stats().backend_drained);
    REQUIRE_FALSE(sequencer.Complete());
    REQUIRE(sequencer.Stats().retired_instructions == 4);
}
TEST_CASE("Illegal programs fail before resource stalls hide them",
          "[instruction-sequencer]") {
    Config hw("configs/HBM2_8Gb_x128_pim.ini", "/tmp");
    auto words = Program();
    SECTION("missing CONFIG") { words = {0xb714000000000000ULL}; }
    SECTION("early END") {
        words.resize(8);
        words.push_back(0xb714000000000001ULL);
    }
    SECTION("wrong opcode where STORE required") {
        words[8] = 0xb713020000000001ULL;
        words.insert(words.begin() + 10, 0);
    }
    SECTION("wrong compute coordinates while compute busy") { words[14] = 64; }
    SECTION("CONFIG replacement") {
        words[13] = 0xb711070000000003ULL;
        words.resize(21, 0);
    }
    SECTION("extra words buffered behind END") {
        words.push_back(0xb714000000000005ULL);
    }
    PimInstructionBuffer buffer(16);
    PimInstructionSequencer sequencer(hw);
    REQUIRE_THROWS_AS(Run(words, buffer, sequencer), std::invalid_argument);
}
TEST_CASE("Refresh freezes execution but not free descriptor dispatch",
          "[instruction-sequencer]") {
    Config hw("configs/HBM2_8Gb_x128_pim.ini", "/tmp");
    PimInstructionBuffer buffer(16);
    PimInstructionSequencer sequencer(hw);
    auto words = Program();
    for (auto word : words) REQUIRE(buffer.PushWord(word));
    PimTickFeedback feedback;
    feedback.in_refresh = true;
    for (int tick = 0; tick < 30; ++tick) {
        buffer.Tick(tick);
        auto result = sequencer.Tick(tick, buffer, feedback);
        REQUIRE(result.events.empty());
    }
    REQUIRE(sequencer.Stats().dispatched_instructions == 3);
    REQUIRE(sequencer.Stats().retired_instructions == 1);
    REQUIRE_FALSE(sequencer.Stats().backend_drained);
}
