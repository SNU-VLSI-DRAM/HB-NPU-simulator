#include <random>

#include "catch.hpp"
#include "pim_instruction_buffer.h"
using namespace dramsim3;

TEST_CASE("One reserved slot assembles all eight CONFIG words",
          "[instruction-buffer]") {
    PimInstructionBuffer buffer(1);
    REQUIRE(buffer.PushWord(0xb711070000000000ULL));
    for (int i = 0; i < 7; ++i) REQUIRE(buffer.PushWord(i));
    for (int i = 0; i < 8; ++i) buffer.Tick(i);
    REQUIRE(buffer.SlotSize() == 1);
    REQUIRE(buffer.RawSize() == 0);
    REQUIRE(buffer.Front(7) == nullptr);
    REQUIRE(buffer.Front(8) != nullptr);
    REQUIRE(buffer.Stats().admitted_words == 8);
    REQUIRE_FALSE(buffer.HasPartialPacket());
    REQUIRE(buffer.Front(8)->payload[6] == 6);
    REQUIRE(buffer.PushWord(0xb714000000000001ULL));
    buffer.Tick(8);
    REQUIRE(buffer.RawSize() == 1);  // occupied slot blocks only header
    buffer.Pop();
    buffer.Tick(9);
    REQUIRE(buffer.Front(9) == nullptr);
    REQUIRE(buffer.Front(10)->header.sequence == 1);
    buffer.Pop();
    REQUIRE(buffer.Empty());
}
TEST_CASE("Raw FIFO retries cannot overwrite or count rejected words",
          "[instruction-buffer]") {
    PimInstructionBuffer buffer(1);
    for (int i = 0; i < 32; ++i)
        REQUIRE(buffer.PushWord(0xb714000000000000ULL | i));
    REQUIRE_FALSE(buffer.PushWord(0));
    REQUIRE(buffer.Stats().admitted_words == 32);
    buffer.Tick(0);
    REQUIRE(buffer.PushWord(0xb714000000000020ULL));
    for (int i = 0; i < 33; ++i) {
        REQUIRE(buffer.Front(i * 2 + 1)->header.sequence == uint32_t(i));
        buffer.Pop();
        buffer.Tick(i * 2 + 2);
    }
    REQUIRE(buffer.Empty());
    REQUIRE(buffer.Stats().decoded_instructions == 33);
}
TEST_CASE("Malformed sequence and capacity fail instead of resynchronizing",
          "[instruction-buffer]") {
    REQUIRE_THROWS(PimInstructionBuffer(0));
    for (uint64_t bad : {0xb714000000000001ULL, 0xb7140000ffffffffULL, 0ULL}) {
        PimInstructionBuffer buffer(1);
        buffer.PushWord(bad);
        REQUIRE_THROWS(buffer.Tick(0));
    }
    PimInstructionBuffer buffer(2);
    buffer.PushWord(0xb714000000000000ULL);
    buffer.PushWord(0xb714000000000000ULL);
    buffer.Tick(0);
    REQUIRE_THROWS(buffer.Tick(1));
}
TEST_CASE("Sparse producers and consumers retain bounded ordered packets",
          "[instruction-buffer]") {
    for (size_t capacity : {1, 16}) {
        PimInstructionBuffer buffer(capacity);
        std::mt19937 random(42);
        uint32_t sent = 0, received = 0;
        for (uint64_t tick = 0; tick < 10000 && received < 300; ++tick) {
            buffer.Tick(tick);
            if (random() % 3 == 0 && buffer.Front(tick)) {
                REQUIRE(buffer.Front(tick)->header.sequence == received++);
                buffer.Pop();
            }
            if (sent < 300 && random() % 2 &&
                buffer.PushWord(0xb714000000000000ULL | sent))
                ++sent;
            REQUIRE(buffer.SlotSize() <= capacity);
            REQUIRE(buffer.RawSize() <= 32);
        }
        REQUIRE(received == 300);
        REQUIRE(buffer.Empty());
    }
}
