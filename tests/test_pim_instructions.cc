#include <climits>
#include <fstream>

#include "catch.hpp"
#include "json.hpp"
#include "hbnpu/pim_instruction.h"

using namespace dramsim3;
namespace {
PimInstructionPacket ConfigPacket() {
    PimInstructionPacket p;
    p.header = PimInstructionCodec::DecodeHeader(0xb711070000000000ULL);
    p.payload = {{0x8000000040ULL, 0x80000000100ULL, 0x855, 0, 0x855, 0x10200,
                  0x0020001000000008ULL}};
    return p;
}
}  // namespace
TEST_CASE("Instruction headers use exact v1 wire format",
          "[instruction-codec]") {
    PimInstructionHeader invalid;
    invalid.opcode = static_cast<PimInstructionOpcode>(17);
    invalid.payload_words = 7;
    REQUIRE_THROWS_AS(PimInstructionCodec::EncodeHeader(invalid),
                      std::invalid_argument);
    std::ifstream in("tests/regression/fixtures/instruction_words.json");
    nlohmann::json fixtures;
    in >> fixtures;
    for (const auto& item : fixtures) {
        auto word =
            std::stoull(item["words"][0].get<std::string>(), nullptr, 16);
        auto h = PimInstructionCodec::DecodeHeader(word);
        REQUIRE(PimInstructionCodec::EncodeHeader(h) == word);
        REQUIRE(h.payload_words + 1 == item["words"].size());
    }
    REQUIRE(PimInstructionCodec::DecodeHeader(0xb714000000000003ULL).sequence ==
            3);
    for (uint64_t bad :
         {0xa711070000000000ULL, 0xb721070000000000ULL, 0xb715000000000000ULL,
          0xb711060000000000ULL, 0xb714000100000003ULL})
        REQUIRE_THROWS_AS(PimInstructionCodec::DecodeHeader(bad),
                          std::invalid_argument);
}
TEST_CASE("Instruction context rejects unsafe geometry before narrowing",
          "[instruction-codec]") {
    Config hw("configs/HBM2_8Gb_x128_pim.ini", "/tmp");
    auto p = ConfigPacket();
    auto context = PimInstructionCodec::DecodeConfig(p, hw);
    REQUIRE(context.m == 64);
    REQUIRE(context.n == 128);
    REQUIRE(context.k == 256);
    REQUIRE(context.dataflow.mc == 2);
    REQUIRE(context.base_row_in == 0x855);
    SECTION("reserved dataflow") { p.payload[5] |= 1ULL << 24; }
    SECTION("topology mismatch") { p.payload[6] ^= 1; }
    SECTION("zero M") { p.payload[0] &= 0xffffffff00000000ULL; }
    SECTION("signed overflow") { p.payload[0] |= 0xffffffffULL; }
    SECTION("address overflow") { p.payload[2] = UINT64_MAX; }
    SECTION("row extent overflow") { p.payload[2] = hw.rows - 1; }
    SECTION("unsupported banks") { hw.banks = 8; }
    SECTION("invalid interleaving") { p.payload[5] = 0x10300; }
    SECTION("offset product overflow") { p.payload[1] = 0x8007fffffffULL; }
    REQUIRE_THROWS_AS(PimInstructionCodec::DecodeConfig(p, hw),
                      std::invalid_argument);
}
TEST_CASE("Received tile origins determine tails not next work",
          "[instruction-codec]") {
    Config hw("configs/HBM2_8Gb_x128_pim.ini", "/tmp");
    auto p = ConfigPacket();
    p.payload[0] = (uint64_t(129) << 32) | 65;
    p.payload[1] = (uint64_t(64) << 32) | 130;
    auto context = PimInstructionCodec::DecodeConfig(p, hw);
    p.header = PimInstructionCodec::DecodeHeader(0xb713020000000002ULL);
    p.payload[0] = (uint64_t(128) << 32) | 64;
    p.payload[1] = 128;
    auto block = PimInstructionCodec::DecodeCompute(p, context, hw);
    REQUIRE(block.m_extent == 1);
    REQUIRE(block.n_extent == 1);
    REQUIRE(block.k_extent == 2);
    REQUIRE(block.last_block);
    REQUIRE(block.final_k);
    REQUIRE_FALSE(block.first_k);
    p.payload[1] |= 1ULL << 32;
    REQUIRE_THROWS(PimInstructionCodec::DecodeCompute(p, context, hw));
    p.header = PimInstructionCodec::DecodeHeader(0xb712010000000003ULL);
    p.payload[0] = (uint64_t(1) << 32) | 64;
    auto out = PimInstructionCodec::DecodeOutput(p, context, hw);
    REQUIRE(out.m_extent == 1);
    REQUIRE(out.last_output);
    p.payload[0] = 1;
    REQUIRE_THROWS(PimInstructionCodec::DecodeOutput(p, context, hw));
}
