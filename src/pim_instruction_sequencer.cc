#include "pim_instruction_sequencer.h"

#include <algorithm>
#include <stdexcept>
#include <string>
namespace dramsim3 {
namespace {
void Require(bool ok, const PimInstructionPacket& p, const char* message) {
    if (!ok)
        throw std::invalid_argument("PIM instruction " +
                                    std::to_string(p.header.sequence) + ": " +
                                    message);
}
}  // namespace
void PimInstructionSequencer::Dispatch(uint64_t cycle,
                                       PimInstructionBuffer& buffer) {
    const auto* p = buffer.Front(cycle);
    if (!p) {
        if (executor_ && !Complete() &&
            ((expected_ == PimInstructionOpcode::TILE_BLOCK &&
              executor_->CanAcceptCompute()) ||
             (expected_ == PimInstructionOpcode::TILE_GEMM &&
              executor_->Snapshot().iw_status == 1) ||
             (expected_ == PimInstructionOpcode::STORE &&
              executor_->CanAcceptOutput()) ||
             (expected_ == PimInstructionOpcode::END &&
              stats_.backend_drained)))
            ++stats_.instruction_starvation_cycles;
        return;
    }
    Require(!Complete(), *p, "instruction after END");
    Require(p->header.opcode == expected_ ||
                (expected_ == PimInstructionOpcode::TILE_BLOCK &&
                 p->header.opcode == PimInstructionOpcode::LOAD_INPUTS),
            *p,
            "opcode violates CONFIG / STORE / TILE or LOAD-GEMM pair / END "
            "profile");
    switch (p->header.opcode) {
        case PimInstructionOpcode::CONFIG: {
            context_.reset(new PimKernelContext(
                PimInstructionCodec::DecodeConfig(*p, config_)));
            executor_.reset(new PimTileExecutor(config_, *context_));
            n_groups_ = (context_->n - 1) / 128 + 1;
            groups_ =
                uint64_t((context_->m - 1) / context_->dataflow.m_tile_size +
                         1) *
                n_groups_;
            output_n_groups_ = context_->dataflow.df ? 1 : n_groups_;
            expected_ = PimInstructionOpcode::STORE;
            ++stats_.retired_instructions;
            break;
        }
        case PimInstructionOpcode::STORE: {
            auto block =
                PimInstructionCodec::DecodeOutput(*p, *context_, config_);
            const auto& d = context_->dataflow;
            const uint64_t tile =
                d.df ? d.m_tile_size / 128 * d.mcf : d.m_tile_size;
            Require(
                uint64_t(block.m_origin) == group_ / output_n_groups_ * tile &&
                    uint64_t(block.n_tile) == group_ % output_n_groups_,
                *p, "output descriptor order");
            if (!executor_->CanAcceptOutput()) {
                ++stats_.output_resource_stall_cycles;
                return;
            }
            executor_->LoadOutput(block);
            output_active_ = true;
            output_sequence_ = p->header.sequence;
            expected_ = PimInstructionOpcode::TILE_BLOCK;
            break;
        }
        case PimInstructionOpcode::LOAD_INPUTS:
        case PimInstructionOpcode::TILE_GEMM:
        case PimInstructionOpcode::TILE_BLOCK: {
            auto block =
                PimInstructionCodec::DecodeCompute(*p, *context_, config_);
            Require(
                uint64_t(block.m_origin) ==
                        group_ / n_groups_ * context_->dataflow.m_tile_size &&
                    uint64_t(block.n_origin) == group_ % n_groups_ * 128 &&
                    block.k_origin == expected_k_,
                *p, "compute descriptor order");
            if (p->header.opcode == PimInstructionOpcode::TILE_GEMM) {
                executor_->StartCompute(block);
            } else {
                if (!executor_->CanAcceptCompute()) {
                    if (p->header.opcode == PimInstructionOpcode::LOAD_INPUTS)
                        ++stats_.load_resource_stall_cycles;
                    else
                        ++stats_.compute_resource_stall_cycles;
                    return;
                }
                if (p->header.opcode == PimInstructionOpcode::LOAD_INPUTS) {
                    executor_->LoadResident(block);
                    load_active_ = true;
                    expected_ = PimInstructionOpcode::TILE_GEMM;
                    break;
                }
                executor_->LoadCompute(block);
            }
            compute_active_ = true;
            compute_sequence_ = p->header.sequence;
            if (block.final_k) {
                expected_k_ = 0;
                ++group_;
                expected_ = group_ == groups_ ? PimInstructionOpcode::END
                                              : PimInstructionOpcode::STORE;
            } else {
                expected_k_ += block.k_extent;
                expected_ = PimInstructionOpcode::TILE_BLOCK;
            }
            break;
        }
        case PimInstructionOpcode::END:
            Require(buffer.RawSize() == 0 && buffer.SlotSize() == 1 &&
                        !buffer.HasPartialPacket(),
                    *p, "extra buffered data after END");
            if (!stats_.backend_drained) return;
            stats_.end_completed = true;
            stats_.end_cycle = cycle;
            ++stats_.retired_instructions;
            break;
    }
    ++stats_.dispatched_instructions;
    buffer.Pop();
}
PimTickResult PimInstructionSequencer::Tick(uint64_t cycle,
                                            PimInstructionBuffer& buffer,
                                            const PimTickFeedback& feedback) {
    Dispatch(cycle, buffer);
    if (!executor_ || stats_.backend_drained) return PimTickResult{};
    auto result = executor_->Tick(cycle, feedback, PimDescriptorSource{});
    if (compute_active_ && (executor_->CanAcceptCompute() || result.finished)) {
        compute_active_ = false;
        ++stats_.retired_instructions;
    }
    for (auto event : result.events) {
        if (event == PimProgress::ResidentTileReady && load_active_) {
            load_active_ = false;
            ++stats_.retired_instructions;
        }
        if (event == PimProgress::OutputBlockIssued) {
            if (!output_active_)
                throw std::logic_error(
                    "PIM output retirement without active instruction");
            output_active_ = false;
            ++stats_.retired_instructions;
        }
    }
    if (result.finished) {
        stats_.backend_drained = true;
        stats_.backend_drain_cycle = cycle;
    }
    return result;
}
}  // namespace dramsim3
