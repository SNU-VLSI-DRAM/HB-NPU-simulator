#include "reference_tile_schedule.h"

#include <algorithm>

namespace dramsim3 {

ReferenceTileSchedule::ReferenceTileSchedule(const PimKernelContext& context, int channels)
    : context_(context), k_tile_size_(std::min(channels * 16, context.k)) {
    context_.Validate(channels);
}

bool ReferenceTileSchedule::NextCompute(PimComputeBlock& block) {
    if (m_origin_ >= context_.m) return false;
    block.m_origin = m_origin_;
    block.n_origin = n_origin_;
    block.k_origin = k_origin_;
    block.m_extent = std::min(context_.dataflow.m_tile_size, context_.m - m_origin_);
    block.n_extent = std::min(128, context_.n - n_origin_);
    block.k_extent = std::min(k_tile_size_, context_.k - k_origin_);
    block.first_k = k_origin_ == 0;
    block.final_k = k_origin_ + block.k_extent == context_.k;
    block.last_block = block.final_k && n_origin_ + block.n_extent == context_.n &&
                       m_origin_ + block.m_extent == context_.m;
    k_origin_ += k_tile_size_;
    if (k_origin_ >= context_.k) {
        k_origin_ = 0;
        n_origin_ += 128;
        if (n_origin_ >= context_.n) {
            n_origin_ = 0;
            m_origin_ += context_.dataflow.m_tile_size;
        }
    }
    return true;
}

bool ReferenceTileSchedule::NextOutput(PimOutputBlock& block) {
    const auto& config = context_.dataflow;
    const int m_tile = config.df == 1 ? (config.m_tile_size / 128) * config.mcf : config.m_tile_size;
    const int m = config.df == 1 ? std::max(1, context_.m * config.mcf / 128) : context_.m;
    const int n = config.df == 1 ? 128 : context_.n;
    if (output_m_origin_ >= m) return false;
    block.m_origin = output_m_origin_;
    block.n_tile = output_n_tile_;
    block.m_extent = std::min(m_tile, m - output_m_origin_);
    block.last_output = output_m_origin_ + block.m_extent == m &&
                        (output_n_tile_ + 1) * 128 >= n;
    ++output_n_tile_;
    if (output_n_tile_ * 128 >= n) {
        output_n_tile_ = 0;
        output_m_origin_ += m_tile;
    }
    return true;
}

void ReferenceTileSchedule::ProjectProgress(PimExecutionState& state,
                                        bool compute_exhausted,
                                        bool output_exhausted) const {
    if (compute_exhausted) {
        state.m_it = m_origin_;
        state.n_it = n_origin_;
        state.k_tile_it = k_origin_ / k_tile_size_;
    }
    if (output_exhausted) {
        state.m_out_it = output_m_origin_;
        state.n_out_tile_it = output_n_tile_;
    }
}

}  // namespace dramsim3
