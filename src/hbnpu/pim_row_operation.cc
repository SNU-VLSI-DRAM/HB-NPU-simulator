#include "hbnpu/pim_row_operation.h"
namespace dramsim3 {
PimRowOperation PimRowOperation::Build(const Config& hw,
                                       const PimKernelContext& c, int m,
                                       int k_tile, int m_end) {
    const auto& d = c.dataflow;
    if (m < 0 || m >= m_end || m_end > c.m || k_tile < 0 ||
        d.m_tile_size <= 0 || hw.BL <= 0 || hw.columns < hw.BL ||
        hw.tCCD_L <= 0 || hw.tCCD_S <= 0 ||
        !((d.df == 0 && d.mc == 2) || (d.df == 1 && d.mc == 16)))
        throw std::invalid_argument("PIM row geometry");
    const int kt = std::min(hw.channels * 16, c.k);
    const int nk = (c.k - 1) / kt + 1;
    const int mt = m / d.m_tile_size;
    const int extent = std::min(d.m_tile_size, c.m - mt * d.m_tile_size);
    if (k_tile >= nk || m_end > mt * d.m_tile_size + extent)
        throw std::invalid_argument("PIM row crosses compute tile");
    const int offset =
        mt * (d.m_tile_size * nk) + k_tile * extent + m % d.m_tile_size;
    const int columns = hw.columns / hw.BL;
    PimRowOperation row;
    row.m_origin = m;
    row.k_tile = k_tile;
    row.vectors = std::min(m_end - m, columns - offset % columns);
    row.operations_per_vector = d.df == 0 ? d.mc : 1;
    row.period =
        std::max({hw.burst_cycle, hw.tCCD_L,
                  row.operations_per_vector * (d.df == 0 ? hw.tCCD_S : 1)});
    row.channels.resize(hw.channels);
    for (int ch = 0; ch < hw.channels; ++ch) {
        for (int v = 0; v < row.vectors; ++v) {
            std::vector<Command> commands;
            for (int slot = 0; slot < d.mc; ++slot) {
                int bank = slot * (hw.banks / d.mc) + (d.df == 0 ? 1 : 0);
                Address addr(ch, 0, bank / hw.banks_per_group,
                             bank % hw.banks_per_group,
                             c.base_row_in + (offset + v) / columns,
                             (offset + v) % columns);
                bool close = m + v + 1 == c.m;
                if (d.df == 0) close &= (k_tile + 1) * kt >= c.k;
                close |= addr.column == columns - 1;
                CommandType type = d.df == 0
                                       ? (close ? CommandType::GH_READ_PRECHARGE
                                                : CommandType::GH_READ)
                                       : (close ? CommandType::LH_READ_PRECHARGE
                                                : CommandType::LH_READ);
                commands.emplace_back(type, addr, hw.AddressUnmapping(addr));
            }
            if (d.df == 1) {
                PimOperation op(commands.front(), v * row.period);
                op.commands = std::move(commands);
                row.channels[ch].push_back(std::move(op));
            } else {
                for (int slot = 0; slot < d.mc; ++slot)
                    row.channels[ch].emplace_back(
                        commands[slot], v * row.period + slot * hw.tCCD_S);
            }
        }
    }
    // Conservative recovery bound, including enough slots to precharge all
    // banks. No refresh is postponed: admission must fit before the existing
    // deadline.
    row.refresh_budget = row.channels.front().back().release_cycle +
                         std::max(hw.tRAS, hw.AL + hw.tRTP) + hw.tRP +
                         hw.banks + 1;
    // Also require a feasible window after refresh, not just before the first
    // refresh. Account conservatively for recovery, a residual tFAW window,
    // activation waves, tRCD, and the coordinator's observation tick.
    const uint64_t activation_span =
        d.df == 1 ? uint64_t((hw.banks + 3) / 4 - 1) * std::max(1, hw.tFAW)
                  : row.operations_per_vector - 1;
    row.restart_budget = row.refresh_budget + std::max(hw.tRFC, hw.tRFCb) +
                         hw.tFAW + activation_span + hw.tRCDRD + 1;
    return row;
}
}  // namespace dramsim3
