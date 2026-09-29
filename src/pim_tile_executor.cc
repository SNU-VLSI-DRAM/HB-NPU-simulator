#include "pim_tile_executor.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace dramsim3 {

PimTileExecutor::PimTileExecutor(const Config& hardware, const PimKernelContext& context)
    : config_(hardware), context_(context), pim_config_(context_.dataflow) {
    context_.Validate(config_.channels);
    state_.m = context_.m; state_.n = context_.n; state_.k = context_.k;
    state_.base_row_in = context_.base_row_in;
    state_.base_row_weight = context_.base_row_weight;
    state_.base_row_output = context_.base_row_output;
    state_.in_pim = true;
}

bool PimTileExecutor::CanAcceptCompute() const {
    return state_.in_pim && state_.iw_status == 0 && !compute_active_;
}

void PimTileExecutor::LoadCompute(const PimComputeBlock& block) {
    if (!CanAcceptCompute())
        throw std::logic_error("PIM compute busy at M/N/K " + std::to_string(compute_.m_origin) +
            "/" + std::to_string(compute_.n_origin) + "/" + std::to_string(compute_.k_origin));
    const int k_tile = std::min(config_.channels * 16, context_.k);
    if (block.m_origin < 0 || block.m_origin >= context_.m ||
        block.n_origin < 0 || block.n_origin >= context_.n ||
        block.k_origin < 0 || block.k_origin >= context_.k ||
        block.m_origin % pim_config_.m_tile_size || block.n_origin % 128 ||
        block.k_origin % k_tile ||
        block.m_extent != std::min(pim_config_.m_tile_size, context_.m - block.m_origin) ||
        block.n_extent != std::min(128, context_.n - block.n_origin) ||
        block.k_extent != std::min(k_tile, context_.k - block.k_origin) ||
        block.first_k != (block.k_origin == 0) ||
        block.final_k != (block.k_extent == context_.k - block.k_origin) ||
        block.last_block != (block.final_k && block.n_extent == context_.n - block.n_origin &&
                              block.m_extent == context_.m - block.m_origin))
        throw std::invalid_argument("PIM compute descriptor geometry or boundary flags");
    compute_ = block;
    compute_active_ = true;
    compute_exhausted_ = false;
    state_.m_it = block.m_origin;
    state_.n_it = block.n_origin;
    state_.k_tile_it = block.k_origin / k_tile;
}

bool PimTileExecutor::CanAcceptOutput() const {
    return state_.in_pim && !output_active_;
}

void PimTileExecutor::LoadOutput(const PimOutputBlock& block) {
    if (!CanAcceptOutput())
        throw std::logic_error("PIM output busy at M/N " + std::to_string(output_.m_origin) +
            "/" + std::to_string(output_.n_tile));
    const int m_tile = pim_config_.df == 1 ?
        (pim_config_.m_tile_size / 128) * pim_config_.mcf : pim_config_.m_tile_size;
    const int m = pim_config_.df == 1 ? std::max(1, context_.m * pim_config_.mcf / 128) : context_.m;
    const int n = pim_config_.df == 1 ? 128 : context_.n;
    if (block.m_origin < 0 || block.m_origin >= m || block.m_origin % m_tile ||
        block.n_tile < 0 || block.n_tile > (n - 1) / 128 ||
        block.m_extent != std::min(m_tile, m - block.m_origin) ||
        block.last_output != (block.m_extent == m - block.m_origin && block.n_tile == (n - 1) / 128))
        throw std::invalid_argument("PIM output descriptor geometry or boundary flags");
    output_ = block;
    output_active_ = true;
    output_exhausted_ = false;
    state_.m_out_it = block.m_origin;
    state_.n_out_tile_it = block.n_tile;
}

PimTickResult PimTileExecutor::Tick(uint64_t clk_, const PimTickFeedback& feedback,
                                  const PimDescriptorSource& source) {
    PimTickResult result;
    result.compute_exhausted = compute_exhausted_;
    result.output_exhausted = output_exhausted_;
    PimExecutionState& state = state_;
    const bool wait_refresh = feedback.wait_refresh;
    if (!pim_config_.configured || !state.in_pim || feedback.in_refresh) return result;
    const int N_tile_size = 128;
    const int K_tile_size = std::min(config_.channels * 16, state.k);
    const int array_period = pim_config_.df == 0 ? config_.tCCD_S : config_.tCCD_L;
    const int weight_banks_reduce = pim_config_.df == 0 ? 8 : 16;
    const int N_tile_size_per_bank = std::min(state.n,
        (N_tile_size-1)/(config_.banks/weight_banks_reduce)+1);
    const bool output_ready = state.iw_status == 3;
    PimCommandBatch& batch = result.batch;
    auto add_vector = [&](PimSource source, const std::vector<Command>& commands) {
        if (commands.empty()) return;
        PimOperation operation(commands.front(),clk_);
        if (operation.IsLocal()) {
            operation.commands = commands;
            batch.Add(source,operation);
        } else {
            for (const auto& cmd : commands) batch.Add(source,PimOperation(cmd,clk_));
        }
    };

    switch (state.iw_status) {
        case 0: {
            if (CanAcceptCompute()) {
                PimComputeBlock next;
                if (source.next_compute && source.next_compute(next)) LoadCompute(next);
            }
            if (!compute_active_) break;
            // Completion means every addressed bank issued its data command,
            // not merely that the vector was accepted or activated.
            if (state.weight_pending && !feedback.weight_pending) {
                state.weight_pending = false;
                result.events.push_back(PimProgress::ResidentVectorIssued);
                int N_tile_it = state.n_it/N_tile_size;
                ++state.n_it;
                if (state.n_it % N_tile_size_per_bank == 0 &&
                    (N_tile_size == N_tile_size_per_bank || state.n_it % N_tile_size != 0)) {
                    state.n_it = N_tile_size*N_tile_it;
                    ++state.iw_status;
                }
            }
            if (state.iw_status != 0 || state.weight_pending || wait_refresh) break;
            int N_tile_it = state.n_it/N_tile_size;
            int col_offset = N_tile_it*(N_tile_size_per_bank*((state.k-1)/K_tile_size+1)) +
                             state.k_tile_it*N_tile_size_per_bank + state.n_it%N_tile_size;
            for (int ch=0; ch<config_.channels; ++ch) {
                std::vector<Command> commands;
                for (int k=0; k<config_.banks/weight_banks_reduce; ++k) {
                    int bank = k*weight_banks_reduce;
                    Address addr(ch,0,bank/config_.banks_per_group,bank%config_.banks_per_group,
                        state.base_row_weight+col_offset/(config_.columns/config_.BL),
                        col_offset%(config_.columns/config_.BL));
                    bool exit = (state.n_it+1)%N_tile_size_per_bank == 0 &&
                        (N_tile_size == N_tile_size_per_bank || (state.n_it+1)%N_tile_size != 0);
                    bool close = (addr.column+1)%std::min(state.n,128/config_.banks*weight_banks_reduce)==0 ||
                                 (addr.column+1)%(config_.columns/config_.BL)==0 || exit;
                    commands.emplace_back(close ? CommandType::GH_READ_PRECHARGE : CommandType::GH_READ,
                                          addr,config_.AddressUnmapping(addr));
                }
                add_vector(PimSource::WEIGHT,commands);
            }
            state.weight_pending = true;
            break;
        }
        case 1:
            ++state.iw_status;
            state.vpu_cnt = 1;
            break;
        case 2: {
            state.vpu_cnt = std::max(0,state.vpu_cnt-1);
            if (state.input_pending && !feedback.input_pending) {
                state.input_pending = false;
                result.events.push_back(PimProgress::InputVectorIssued);
                // Retain the approved array geometry and activation overlap.
                // Trigger these counters on completed data vectors, never PRE.
                if (compute_.final_k && state.m_it == compute_.m_origin)
                    state.out_cnt = std::max(1,array_period*(3+16)-config_.tRCDWR);
                ++state.m_it;
                if (state.m_it == compute_.m_origin + compute_.m_extent) {
                    result.events.push_back(PimProgress::BlockStreamingExhausted);
                    state.in_cnt = std::max(1,array_period*std::max(128/pim_config_.mc,16)-config_.tRCDRD);
                    ++state.iw_status;
                    compute_exhausted_ = true;
                    if (compute_.last_block) {
                        result.computation_finished = true;
                        state.in_cnt = -1;
                    }
                }
            }
            if (state.iw_status != 2 || state.input_pending || wait_refresh || state.vpu_cnt) break;
            int M_tile_it = state.m_it/pim_config_.m_tile_size;
            int M_current_tile_size = state.m < pim_config_.m_tile_size*(M_tile_it+1) ?
                                      state.m%pim_config_.m_tile_size : pim_config_.m_tile_size;
            int col_offset = M_tile_it*(pim_config_.m_tile_size*((state.k-1)/K_tile_size+1)) +
                             state.k_tile_it*M_current_tile_size + state.m_it%pim_config_.m_tile_size;
            for (int ch=0; ch<config_.channels; ++ch) {
                std::vector<Command> commands;
                for (int k=0; k<pim_config_.mc; ++k) {
                    int bank = k*(config_.banks/pim_config_.mc) + (pim_config_.df == 0 ? 1 : 0);
                    Address addr(ch,0,bank/config_.banks_per_group,bank%config_.banks_per_group,
                        state.base_row_in+col_offset/(config_.columns/config_.BL),
                        col_offset%(config_.columns/config_.BL));
                    bool close = state.m_it+1 == state.m;
                    if (pim_config_.df == 0) close &= (state.k_tile_it+1)*K_tile_size >= state.k;
                    close |= addr.column == config_.columns/config_.BL-1;
                    CommandType type = pim_config_.df == 0 ?
                        (close ? CommandType::GH_READ_PRECHARGE : CommandType::GH_READ) :
                        (close ? CommandType::LH_READ_PRECHARGE : CommandType::LH_READ);
                    commands.emplace_back(type,addr,config_.AddressUnmapping(addr));
                }
                add_vector(PimSource::INPUT,commands);
            }
            state.input_pending = true;
            break;
        }
        case 3:
            if (state.in_cnt != -1) {
                state.in_cnt = std::max(0,state.in_cnt-1);
                if (state.in_cnt == 0 && state.output_valid == 0) {
                    state.iw_status = 0;
                    compute_active_ = false;
                }
            }
            break;
        default: break;
    }

    if (state.out_cnt == 0) {
        ++state.output_valid;
        result.events.push_back(PimProgress::OutputStageReleased);
    }
    if (state.out_cnt != -1) --state.out_cnt;

    if (state.output_valid > 0 && output_ready) {
        const int M_tile_size_out = pim_config_.df == 1 ?
            (pim_config_.m_tile_size/128)*pim_config_.mcf : pim_config_.m_tile_size;
        const int M_out = pim_config_.df == 1 ? std::max(1,state.m*pim_config_.mcf/128) : state.m;
        const int N_tile_size_out = 128;
        const int N_tile_num = (state.n-1)/N_tile_size_out+1;
        if (state.output_pending && !feedback.output_pending) {
            state.output_pending = false;
            ++state.m_out_it;
            if (state.m_out_it == output_.m_origin + output_.m_extent) {
                output_active_ = false;
                output_exhausted_ = true;
                if (output_.last_output) {
                    if (state.in_cnt != -1 || !compute_exhausted_ ||
                        feedback.input_pending || feedback.weight_pending)
                        throw std::logic_error("PIM final output issued before compute drained");
                    state.in_pim = false;
                    result.finished = true;
                    result.events.push_back(PimProgress::PipelineDrained);
                }
                --state.output_valid;
                result.events.push_back(PimProgress::OutputBlockIssued);
            }
        }
        if (state.in_pim && state.output_valid > 0 && !state.output_pending && !wait_refresh) {
            if (CanAcceptOutput()) {
                PimOutputBlock next;
                if (source.next_output && source.next_output(next)) LoadOutput(next);
            }
            if (output_active_) {
                int M_out_tile_it = state.m_out_it/M_tile_size_out;
                int M_out_current_tile_size = M_out < M_tile_size_out*(M_out_tile_it+1) ?
                    M_out%M_tile_size_out : M_tile_size_out;
                int col_offset = M_out_tile_it*(M_tile_size_out*N_tile_num) +
                    state.n_out_tile_it*M_out_current_tile_size + state.m_out_it%M_tile_size_out;
                for (int ch=0; ch<config_.channels; ++ch) {
                    std::vector<Command> commands;
                    int count = pim_config_.df == 1 ? 1 : pim_config_.mc;
                    for (int k=0; k<count; ++k) {
                        int bank = k*(config_.banks/pim_config_.mc) + (pim_config_.df == 0 ? 1 : 0);
                        int group = bank/config_.banks_per_group;
                        bank = bank%config_.banks_per_group + (pim_config_.df == 0 ? 2 : 0);
                        Address addr(ch,0,group,bank,
                            state.base_row_output+col_offset/(config_.columns/config_.BL),
                            col_offset%(config_.columns/config_.BL));
                        bool close = state.m_out_it+1 == M_out || addr.column == config_.columns/config_.BL-1;
                        commands.emplace_back(close ? CommandType::PIM_WRITE_PRECHARGE : CommandType::PIM_WRITE,
                                              addr,config_.AddressUnmapping(addr));
                    }
                    add_vector(PimSource::OUTPUT,commands);
                }
                state.output_pending = true;
            }
        }
    }
    result.compute_exhausted = compute_exhausted_;
    result.output_exhausted = output_exhausted_;
    return result;
}

}  // namespace dramsim3
