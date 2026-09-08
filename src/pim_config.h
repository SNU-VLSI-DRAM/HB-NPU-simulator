#ifndef __PIM_CONFIG_H
#define __PIM_CONFIG_H

#include "pim_transaction.h"

namespace dramsim3 {

struct PimDataflowConfig {
    bool configured = false;
    int mcf = 1;
    int ucf = 1;
    int mc = 1;
    int df = -1;
    int m_tile_size = 0;
    int stride = 0;
    int kernel_size = 0;

    void Load(const DecodedPimTransaction& decoded) {
        mcf = decoded.mcf;
        ucf = decoded.ucf;
        df = decoded.df;
        mc = mcf * ucf;
        m_tile_size = decoded.m_tile_size;
        kernel_size = decoded.kernel_size;
        stride = decoded.stride;
        configured = true;
    }
};

}  // namespace dramsim3
#endif  // __PIM_CONFIG_H
