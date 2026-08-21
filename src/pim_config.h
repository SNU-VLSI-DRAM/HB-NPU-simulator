#ifndef __PIM_CONFIG_H
#define __PIM_CONFIG_H

#include "pim_transaction.h"

namespace dramsim3 {

struct PimDataflowConfig {
    int vcuts = -1;
    int hcuts = -1;
    int mcf = 1;
    int ucf = 1;
    int mc = 1;
    int df = -1;
    int vcuts_next = -1;
    int hcuts_next = -1;
    int m_tile_size = 0;
    int stride = 0;
    int kernel_size = 0;

    void LoadLayout(const DecodedPimTransaction& decoded) {
        vcuts = decoded.vcuts;
        hcuts = decoded.hcuts;
        mcf = decoded.mcf;
        ucf = decoded.ucf;
        df = decoded.df;
        mc = mcf * ucf;
    }

    void LoadTiling(const DecodedPimTransaction& decoded) {
        m_tile_size = decoded.m_tile_size;
        vcuts_next = decoded.vcuts_next;
        hcuts_next = decoded.hcuts_next;
        kernel_size = decoded.kernel_size;
        stride = decoded.stride;
    }

    void Load(const DecodedPimTransaction& decoded) {
        LoadLayout(decoded);
        LoadTiling(decoded);
    }

    int CutCount() const { return vcuts * hcuts; }
};

}  // namespace dramsim3
#endif  // __PIM_CONFIG_H
