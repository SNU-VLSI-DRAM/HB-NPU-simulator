#ifndef DRAMSIM3_PIM_INSTRUCTION_H
#define DRAMSIM3_PIM_INSTRUCTION_H
#include <array>
#include "configuration.h"
#include "pim_kernel_context.h"

namespace dramsim3 {
enum class PimInstructionOpcode : uint8_t { CONFIG=1, STORE=2, TILE_BLOCK=3, END=4 };
struct PimInstructionHeader {
    PimInstructionOpcode opcode = PimInstructionOpcode::END;
    uint8_t payload_words = 0;
    uint32_t sequence = 0;
};
struct PimInstructionPacket {
    PimInstructionHeader header;
    std::array<uint64_t,7> payload{};
    uint64_t eligible_cycle = 0;
};
class PimInstructionCodec {
 public:
    static PimInstructionHeader DecodeHeader(uint64_t);
    static uint64_t EncodeHeader(const PimInstructionHeader&);
    static PimKernelContext DecodeConfig(const PimInstructionPacket&, const Config&);
    static PimComputeBlock DecodeCompute(const PimInstructionPacket&, const PimKernelContext&, const Config&);
    static PimOutputBlock DecodeOutput(const PimInstructionPacket&, const PimKernelContext&, const Config&);
};
}  // namespace dramsim3
#endif
