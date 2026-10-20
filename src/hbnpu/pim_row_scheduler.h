#ifndef DRAMSIM3_PIM_ROW_SCHEDULER_H
#define DRAMSIM3_PIM_ROW_SCHEDULER_H
#include <fstream>

#include "controller.h"
#include "hbnpu/pim_row_operation.h"
namespace dramsim3 {
struct PimRowStats {
    uint64_t launched = 0, completed = 0, boundary_wait_cycles = 0;
    uint64_t last_start = 0, last_end = 0;
};
class PimRowScheduler {
   public:
    PimRowScheduler(const Config&, const std::vector<Controller*>&);
    void Submit(PimRowOperation row, uint64_t cycle);
    void Tick(uint64_t cycle);
    int CompletedVectors() const { return completed_vectors_; }
    bool Running() const { return running_; }
    const PimRowStats& Stats() const { return stats_; }

   private:
    const std::vector<Controller*>& controllers_;
    PimRowOperation row_;
    PimRowStats stats_;
    bool pending_ = false, running_ = false;
    int completed_vectors_ = 0;
    uint64_t requested_ = 0;
    std::ofstream trace_;
};
}  // namespace dramsim3
#endif
