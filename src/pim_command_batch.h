#ifndef __PIM_COMMAND_BATCH_H
#define __PIM_COMMAND_BATCH_H

#include <vector>

#include "common.h"

namespace dramsim3 {

struct PimCommandBatch {
    std::vector<Command> weight_commands;
    std::vector<Command> input_commands;
    std::vector<int> input_release_times;
    std::vector<Command> output_commands;

    void AddWeight(const Command& command) { weight_commands.push_back(command); }
    void AddInput(const Command& command, int release_time) {
        input_commands.push_back(command);
        input_release_times.push_back(release_time);
    }
    void AddOutput(const Command& command) { output_commands.push_back(command); }
    void ClearWeight() { weight_commands.clear(); }
    void ClearInput() {
        input_commands.clear();
        input_release_times.clear();
    }
    void ClearOutput() { output_commands.clear(); }
};

}  // namespace dramsim3
#endif  // __PIM_COMMAND_BATCH_H
