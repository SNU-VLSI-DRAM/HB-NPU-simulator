#include "pim_transaction.h"

namespace dramsim3 {
namespace {

uint64_t TakeBits(uint64_t& value, int width) {
    uint64_t mask = (1ULL << width) - 1;
    uint64_t result = value & mask;
    value >>= width;
    return result;
}

}  // namespace

DecodedPimTransaction PimTransactionDecoder::Decode(uint64_t address) {
    DecodedPimTransaction decoded;
    if (address & 1ULL) {
        decoded.kind = PimTransactionKind::START_COMPUTATION;
        return decoded;
    }

    if ((address & 0x6ULL) == 0x6ULL) {
        decoded.kind = PimTransactionKind::LOAD_DATAFLOW_CONFIG;
        uint64_t payload = address;
        TakeBits(payload, 1);
        TakeBits(payload, 2);
        decoded.mcf = 1 << TakeBits(payload, 3);
        decoded.ucf = 1 << TakeBits(payload, 3);
        decoded.df = static_cast<int>(TakeBits(payload, 1));
        decoded.m_tile_size = 1 << TakeBits(payload, 4);
        decoded.kernel_size = static_cast<int>(TakeBits(payload, 5));
        decoded.stride = static_cast<int>(TakeBits(payload, 5));
        return decoded;
    }

    decoded.kind = PimTransactionKind::LOAD_WORKLOAD_CONFIG;
    uint64_t payload = address;
    TakeBits(payload, 1);
    decoded.load_type = static_cast<int>(TakeBits(payload, 2));
    decoded.dim_value = static_cast<int>(TakeBits(payload, 32));
    decoded.base_row = TakeBits(payload, 22);
    return decoded;
}

}  // namespace dramsim3
