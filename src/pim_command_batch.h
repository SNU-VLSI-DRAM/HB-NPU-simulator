#ifndef __PIM_COMMAND_BATCH_H
#define __PIM_COMMAND_BATCH_H

#include <vector>

#include "common.h"
#include "pim_operation.h"

namespace dramsim3 {

struct PimCommandBatch {
    std::vector<std::pair<PimSource,PimOperation>> operations;
    void Add(PimSource source, const PimOperation& operation) {
        operations.emplace_back(source,operation);
    }
};

}  // namespace dramsim3
#endif  // __PIM_COMMAND_BATCH_H
