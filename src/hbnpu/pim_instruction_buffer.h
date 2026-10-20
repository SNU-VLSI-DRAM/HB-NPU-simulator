#ifndef DRAMSIM3_PIM_INSTRUCTION_BUFFER_H
#define DRAMSIM3_PIM_INSTRUCTION_BUFFER_H
#include <deque>

#include "hbnpu/pim_instruction.h"
namespace dramsim3 {
struct PimIngressStats {
    uint64_t admitted_words = 0, decoded_instructions = 0,
             raw_high_watermark = 0, slot_high_watermark = 0,
             partial_packet_cycles = 0;
};
class PimInstructionBuffer {
   public:
    explicit PimInstructionBuffer(size_t capacity);
    bool CanAcceptWord() const { return raw_.size() < 32; }
    bool PushWord(uint64_t word);
    void Tick(uint64_t cycle);
    const PimInstructionPacket* Front(uint64_t cycle) const;
    void Pop();
    bool Empty() const { return raw_.empty() && slots_.empty(); }
    bool HasPartialPacket() const { return partial_; }
    size_t RawSize() const { return raw_.size(); }
    size_t SlotSize() const { return slots_.size(); }
    const PimIngressStats& Stats() const { return stats_; }

   private:
    const size_t capacity_;
    std::deque<uint64_t> raw_;
    std::deque<PimInstructionPacket> slots_;
    bool partial_ = false;
    uint8_t received_ = 0;
    uint64_t next_sequence_ = 0;
    PimIngressStats stats_;
};
}  // namespace dramsim3
#endif
