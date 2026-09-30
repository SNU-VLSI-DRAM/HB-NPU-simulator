#include "pim_instruction_buffer.h"
#include <algorithm>
#include <stdexcept>
#include <string>
namespace dramsim3 {
PimInstructionBuffer::PimInstructionBuffer(size_t capacity) : capacity_(capacity) {
    if (!capacity) throw std::invalid_argument("PIM instruction capacity must be positive");
}
bool PimInstructionBuffer::PushWord(uint64_t word) {
    if (!CanAcceptWord()) return false;
    raw_.push_back(word); ++stats_.admitted_words;
    stats_.raw_high_watermark=std::max<uint64_t>(stats_.raw_high_watermark,raw_.size());
    return true;
}
void PimInstructionBuffer::Tick(uint64_t cycle) {
    if (partial_) ++stats_.partial_packet_cycles;
    if (raw_.empty() || (!partial_ && slots_.size()==capacity_)) return;
    if (!partial_) {
        auto header=PimInstructionCodec::DecodeHeader(raw_.front());
        if (next_sequence_>UINT32_MAX || header.sequence!=next_sequence_)
            throw std::invalid_argument("PIM instruction sequence " + std::to_string(header.sequence) +
                ": expected " + std::to_string(next_sequence_));
        ++next_sequence_;
        slots_.emplace_back(); slots_.back().header=header;
        received_=0; partial_=true;
        stats_.slot_high_watermark=std::max<uint64_t>(stats_.slot_high_watermark,slots_.size());
    } else {
        slots_.back().payload[received_++]=raw_.front();
    }
    raw_.pop_front();
    if (received_==slots_.back().header.payload_words) {
        if (cycle==UINT64_MAX) throw std::overflow_error("PIM instruction eligibility cycle overflow");
        slots_.back().eligible_cycle=cycle+1;
        partial_=false; ++stats_.decoded_instructions;
    }
}
const PimInstructionPacket* PimInstructionBuffer::Front(uint64_t cycle) const {
    if (slots_.empty() || (slots_.size()==1 && partial_) || slots_.front().eligible_cycle>cycle)
        return nullptr;
    return &slots_.front();
}
void PimInstructionBuffer::Pop() {
    if (slots_.empty() || (slots_.size()==1 && partial_))
        throw std::logic_error("PIM instruction pop without a complete packet");
    slots_.pop_front();
}
}
