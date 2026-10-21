#ifndef __PIM_CONFIG_H
#define __PIM_CONFIG_H

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

};

}  // namespace dramsim3
#endif  // __PIM_CONFIG_H
