#ifndef DRAMSIM3_PIM_ROW_OPERATION_H
#define DRAMSIM3_PIM_ROW_OPERATION_H
#include "hbnpu/pim_kernel_context.h"
#include "hbnpu/pim_operation.h"
namespace dramsim3 {
// A bounded internal instruction: release_cycle is an offset from common START.
// Command metadata is buffered, not operand data. All channels share a
// schedule.
struct PimRowOperation {
    int vectors = 0, operations_per_vector = 0, m_origin = 0, k_tile = 0;
    uint64_t period = 0, refresh_budget = 0, restart_budget = 0;
    std::vector<std::vector<PimOperation>> channels;
    static PimRowOperation Build(const Config&, const PimKernelContext&,
                                 int m_origin, int k_tile, int m_end);
};
}  // namespace dramsim3
#endif
