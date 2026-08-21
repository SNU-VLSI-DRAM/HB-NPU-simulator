#ifndef __PIM_PARTITION_STATE_H
#define __PIM_PARTITION_STATE_H

#include <stdint.h>

namespace dramsim3 {

struct PimPartitionState {
    uint64_t base_row_in = 0;
    uint64_t base_row_weight = 0;
    uint64_t base_row_output = 0;
    int m = 0;
    int n = 0;
    int k = 0;
    int m_it = 0;
    int n_it = 0;
    int k_tile_it = 0;
    int m_out_it = 0;
    int n_out_tile_it = 0;
    bool in_pim = false;
    int iw_status = 0;
    bool in_act_placed = false;
    bool weight_act_placed = false;
    bool output_act_placed = false;
    int output_valid = 0;
    int in_cnt = 0;
    int out_cnt = -1;
    int vpu_cnt = 0;
};

}  // namespace dramsim3
#endif  // __PIM_PARTITION_STATE_H
