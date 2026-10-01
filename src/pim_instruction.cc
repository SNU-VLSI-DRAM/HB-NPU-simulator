#include "pim_instruction.h"

#include <algorithm>
#include <climits>
#include <stdexcept>
#include <string>

namespace dramsim3 {
namespace {
void Check(bool ok, uint32_t sequence, const char* field) {
    if (!ok)
        throw std::invalid_argument("PIM instruction " +
                                    std::to_string(sequence) + ": " + field);
}
bool Power(uint64_t n) { return n && !(n & (n - 1)); }
uint64_t Ceil(uint64_t n, uint64_t d) { return (n - 1) / d + 1; }
int Signed(uint64_t value, uint32_t seq, const char* name) {
    Check(value <= INT_MAX, seq, name);
    return static_cast<int>(value);
}
void Kind(const PimInstructionPacket& p, PimInstructionOpcode kind) {
    PimInstructionCodec::EncodeHeader(
        p.header);  // also validates payload length
    Check(p.header.opcode == kind, p.header.sequence, "opcode");
}
}  // namespace
PimInstructionHeader PimInstructionCodec::DecodeHeader(uint64_t word) {
    PimInstructionHeader h;
    h.sequence = static_cast<uint32_t>(word);
    Check((word >> 56) == 0xb7, h.sequence, "magic");
    Check(((word >> 52) & 15) == 1, h.sequence, "version");
    const unsigned op = (word >> 48) & 15;
    Check(op >= 1 && op <= 6, h.sequence, "opcode");
    h.opcode = static_cast<PimInstructionOpcode>(op);
    h.payload_words = (word >> 40) & 255;
    const unsigned lengths[] = {0, 7, 1, 2, 0, 2, 2};
    Check(h.payload_words == lengths[op], h.sequence, "payload length");
    Check(((word >> 32) & 255) == 0, h.sequence, "reserved header bits");
    return h;
}
uint64_t PimInstructionCodec::EncodeHeader(const PimInstructionHeader& h) {
    Check(static_cast<unsigned>(h.opcode) >= 1 &&
              static_cast<unsigned>(h.opcode) <= 6,
          h.sequence, "opcode");
    uint64_t word = 0xb710000000000000ULL | (uint64_t(h.opcode) << 48) |
                    (uint64_t(h.payload_words) << 40) | h.sequence;
    DecodeHeader(word);
    return word;
}
PimKernelContext PimInstructionCodec::DecodeConfig(
    const PimInstructionPacket& p, const Config& hw) {
    Kind(p, PimInstructionOpcode::CONFIG);
    auto s = p.header.sequence;
    PimKernelContext c;
    c.m = Signed(uint32_t(p.payload[0]), s, "M");
    c.n = Signed(p.payload[0] >> 32, s, "N");
    c.k = Signed(uint32_t(p.payload[1]), s, "K");
    auto& d = c.dataflow;
    d.m_tile_size = Signed(p.payload[1] >> 32, s, "M tile");
    c.base_row_in = p.payload[2];
    c.base_row_weight = p.payload[3];
    c.base_row_output = p.payload[4];
    Check((p.payload[5] >> 24) == 0, s, "reserved CONFIG bits");
    d.df = p.payload[5] & 255;
    d.mcf = (p.payload[5] >> 8) & 255;
    d.ucf = (p.payload[5] >> 16) & 255;
    d.mc = d.mcf * d.ucf;
    d.configured = true;
    Check(hw.channels > 0 && hw.channels <= INT_MAX / 16 && hw.banks == 16 &&
              hw.banks_per_group == 4 && hw.bankgroups == 4 && hw.BL > 0 &&
              hw.columns >= hw.BL && hw.columns % hw.BL == 0 && hw.rows > 0,
          s, "hardware geometry");
    Check(uint32_t(p.payload[6]) == uint32_t(hw.channels) &&
              ((p.payload[6] >> 32) & 65535) == uint64_t(hw.banks) &&
              (p.payload[6] >> 48) == uint64_t(hw.columns / hw.BL),
          s, "topology mismatch");
    Check(c.m > 0 && c.n > 0 && c.k > 0 && Power(d.m_tile_size), s,
          "positive dimensions and power-of-two M tile");
    Check(Power(d.mcf) && Power(d.ucf) &&
              ((d.df == 0 && d.mc == 2) || (d.df == 1 && d.mc == 16)),
          s, "dataflow/interleaving geometry");
    const uint64_t mt = d.m_tile_size, kt = std::min(hw.channels * 16, c.k);
    const uint64_t nk = Ceil(c.k, kt), nn = Ceil(c.n, 128), nm = Ceil(c.m, mt);
    Check(uint64_t(c.m) * d.mcf <= INT_MAX, s, "output M product");
    const uint64_t mo =
        d.df ? std::max(uint64_t(1), uint64_t(c.m) * d.mcf / 128) : c.m;
    const uint64_t ot = d.df ? mt / 128 * d.mcf : mt;
    Check(ot > 0 && ot <= INT_MAX, s, "output tile");
    Check(nm * nn == Ceil(mo, ot) * (d.df ? 1 : nn), s,
          "output/compute group count");
    const uint64_t nb = std::min(c.n, d.df ? 128 : 64);
    // Bound the padded layouts as well as every signed executor intermediate.
    const uint64_t input_columns = nm * mt * nk, weight_columns = nn * nb * nk;
    const uint64_t output_columns = Ceil(mo, ot) * ot * nn;
    Check(nm * mt <= INT_MAX && nk * kt <= INT_MAX && nn * 128 <= INT_MAX &&
              input_columns <= INT_MAX && weight_columns <= INT_MAX &&
              output_columns <= INT_MAX,
          s, "address offset product");
    Check(hw.tCCD_S > 0 && hw.tCCD_L > 0 &&
              uint64_t(std::max(hw.tCCD_S, hw.tCCD_L)) * 128 <= INT_MAX,
          s, "array period");
    const uint64_t bases[] = {c.base_row_in, c.base_row_weight,
                              c.base_row_output};
    const uint64_t spans[] = {input_columns, weight_columns, output_columns};
    for (int i = 0; i < 3; ++i) {
        Check(bases[i] < uint64_t(hw.rows), s, "base row");
        Check((spans[i] - 1) / (hw.columns / hw.BL) <
                  uint64_t(hw.rows) - bases[i],
              s, "row extent");
    }
    return c;
}
PimComputeBlock PimInstructionCodec::DecodeCompute(
    const PimInstructionPacket& p, const PimKernelContext& c,
    const Config& hw) {
    EncodeHeader(p.header);
    Check(p.header.opcode == PimInstructionOpcode::TILE_BLOCK ||
              p.header.opcode == PimInstructionOpcode::LOAD_INPUTS ||
              p.header.opcode == PimInstructionOpcode::TILE_GEMM,
          p.header.sequence, "tile opcode");
    auto s = p.header.sequence;
    Check((p.payload[1] >> 32) == 0, s, "reserved TILE bits");
    PimComputeBlock b;
    b.m_origin = Signed(uint32_t(p.payload[0]), s, "tile M");
    b.n_origin = Signed(p.payload[0] >> 32, s, "tile N");
    b.k_origin = Signed(p.payload[1], s, "tile K");
    const int kt = std::min(hw.channels * 16, c.k);
    Check(b.m_origin < c.m && b.n_origin < c.n && b.k_origin < c.k &&
              b.m_origin % c.dataflow.m_tile_size == 0 &&
              b.n_origin % 128 == 0 && b.k_origin % kt == 0,
          s, "tile origin");
    b.m_extent = std::min(c.dataflow.m_tile_size, c.m - b.m_origin);
    b.n_extent = std::min(128, c.n - b.n_origin);
    b.k_extent = std::min(kt, c.k - b.k_origin);
    b.first_k = b.k_origin == 0;
    b.final_k = b.k_extent == c.k - b.k_origin;
    b.last_block = b.final_k && b.m_extent == c.m - b.m_origin &&
                   b.n_extent == c.n - b.n_origin;
    return b;
}
PimOutputBlock PimInstructionCodec::DecodeOutput(const PimInstructionPacket& p,
                                                 const PimKernelContext& c,
                                                 const Config&) {
    Kind(p, PimInstructionOpcode::STORE);
    const auto& d = c.dataflow;
    const int mt = d.df ? d.m_tile_size / 128 * d.mcf : d.m_tile_size;
    const int m = d.df ? std::max(1, c.m * d.mcf / 128) : c.m;
    const int n = d.df ? 128 : c.n;
    PimOutputBlock b;
    b.m_origin = Signed(uint32_t(p.payload[0]), p.header.sequence, "output M");
    b.n_tile = Signed(p.payload[0] >> 32, p.header.sequence, "output N");
    Check(b.m_origin < m && b.m_origin % mt == 0 && b.n_tile <= (n - 1) / 128,
          p.header.sequence, "output origin");
    b.m_extent = std::min(mt, m - b.m_origin);
    b.last_output = b.m_extent == m - b.m_origin && b.n_tile == (n - 1) / 128;
    return b;
}
}  // namespace dramsim3
