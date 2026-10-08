#ifndef __REFRESH_H
#define __REFRESH_H

#include <vector>
#include "channel_state.h"
#include "common.h"
#include "configuration.h"

namespace dramsim3 {

class Refresh {
   public:
    Refresh(const Config& config, ChannelState& channel_state);
    void ClockTick();
    bool pim_refresh_coming();
    bool pim_refresh_coming2();
    uint64_t Interval() const { return refresh_interval_; }
    uint64_t CyclesToDeadline() const {
        return clk_ && clk_ % refresh_interval_ == 0 ? 0 :
            refresh_interval_ - clk_ % refresh_interval_;
    }

   private:
    uint64_t clk_;
    int refresh_interval_;
    const Config& config_;
    ChannelState& channel_state_;
    RefreshPolicy refresh_policy_;

    int next_rank_, next_bg_, next_bank_;

    void InsertRefresh();

    void IterateNext();
};

}  // namespace dramsim3

#endif
