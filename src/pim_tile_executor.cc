#include "pim_tile_executor.h"

#include <algorithm>
#include <cassert>
#include <iostream>

namespace dramsim3 {

PimTileExecutor::PimTileExecutor(const Config& hardware, const PimKernelContext& context)
    : config_(hardware), context_(context), pim_config_(context_.dataflow) {
    context_.Validate(config_.channels);
}

PimTickResult PimTileExecutor::RunLegacy(uint64_t clk_, const PimTickFeedback& feedback,
                                        PimExecutionState& state) {
    PimTickResult result;
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
                int M_tile_it = state.m_it/pim_config_.m_tile_size;
                int N_tile_it = state.n_it/N_tile_size;
                // Retain the approved array geometry and activation overlap.
                // Trigger these counters on completed data vectors, never PRE.
                if ((state.k_tile_it+1)*K_tile_size >= state.k &&
                    state.m_it%pim_config_.m_tile_size == 0)
                    state.out_cnt = std::max(1,array_period*(3+16)-config_.tRCDWR);
                ++state.m_it;
                if (state.m_it%pim_config_.m_tile_size == 0 || state.m_it == state.m) {
                    result.events.push_back(PimProgress::BlockStreamingExhausted);
                    state.in_cnt = std::max(1,array_period*std::max(128/pim_config_.mc,16)-config_.tRCDRD);
                    ++state.iw_status;
                    state.m_it = pim_config_.m_tile_size*M_tile_it;
                    ++state.k_tile_it;
                    if (state.k_tile_it*K_tile_size >= state.k) {
                        state.k_tile_it = 0;
                        state.n_it = N_tile_size*(N_tile_it+1);
                        if (state.n_it >= state.n) {
                            state.n_it = 0;
                            state.m_it = pim_config_.m_tile_size*(M_tile_it+1);
                            if (state.m_it >= state.m) {
                                std::cout << clk_ << " End of Computation 0" << std::endl;
                                state.in_cnt = -1;
                            }
                        }
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
                if (state.in_cnt == 0 && state.output_valid == 0) state.iw_status = 0;
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
        const int N_out = pim_config_.df == 1 ? 128 : state.n;
        const int N_tile_size_out = 128;
        const int N_tile_num = (state.n-1)/N_tile_size_out+1;
        if (state.output_pending && !feedback.output_pending) {
            state.output_pending = false;
            int M_out_tile_it = state.m_out_it/M_tile_size_out;
            ++state.m_out_it;
            if (state.m_out_it%M_tile_size_out == 0 || state.m_out_it == M_out) {
                state.m_out_it = M_tile_size_out*M_out_tile_it;
                ++state.n_out_tile_it;
                if (state.n_out_tile_it*N_tile_size_out >= N_out) {
                    state.n_out_tile_it = 0;
                    state.m_out_it = M_tile_size_out*(M_out_tile_it+1);
                    if (state.m_out_it >= M_out) {
                        assert(state.in_cnt == -1);
                        assert(!feedback.input_pending && !feedback.weight_pending);
                        std::cout << clk_ << " Output Exhausted: Array0. Turn off PIM mode.\n";
                        state.in_pim = false;
                        result.finished = true;
                        result.events.push_back(PimProgress::PipelineDrained);
                    }
                }
                --state.output_valid;
                result.events.push_back(PimProgress::OutputBlockIssued);
            }
        }
        if (state.in_pim && state.output_valid > 0 && !state.output_pending && !wait_refresh) {
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
    return result;
}

}  // namespace dramsim3
