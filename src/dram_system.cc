#include "dram_system.h"
#include "hbnpu/pim_transaction.h"
#include <assert.h>
#include <cmath>
#include "json.hpp"
namespace dramsim3 {

// alternative way is to assign the id in constructor but this is less
// destructive
int BaseDRAMSystem::total_channels_ = 0;

BaseDRAMSystem::BaseDRAMSystem(Config &config, const std::string &output_dir,
                               std::function<void(uint64_t)> read_callback,
                               std::function<void(uint64_t)> write_callback)
    : read_callback_(read_callback),
      write_callback_(write_callback),
      last_req_clk_(0),
      config_(config),
      timing_(config_),
#ifdef THERMAL
      thermal_calc_(config_),
#endif  // THERMAL
      clk_(0) {
    total_channels_ += config_.channels;

    // std::cout<<config_.ranks<<config_.banks_per_group<<config_.bankgroups<<std::endl;
#ifdef ADDR_TRACE
    std::string addr_trace_name = config_.output_prefix + "addr.trace";
    address_trace_.open(addr_trace_name);
#endif
}

int BaseDRAMSystem::GetChannel(uint64_t hex_addr) const {
    hex_addr >>= config_.shift_bits;
    return (hex_addr >> config_.ch_pos) & config_.ch_mask;
}

void BaseDRAMSystem::PrintEpochStats() {
    // first epoch, print bracket
    if (clk_ - config_.epoch_period == 0) {
        std::ofstream epoch_out(config_.json_epoch_name, std::ofstream::out);
        epoch_out << "[";
    }
    for (size_t i = 0; i < ctrls_.size(); i++) {
        ctrls_[i]->PrintEpochStats();
        std::ofstream epoch_out(config_.json_epoch_name, std::ofstream::app);
        epoch_out << "," << std::endl;
    }
#ifdef THERMAL
    thermal_calc_.PrintTransPT(clk_);
#endif  // THERMAL
    return;
}

void BaseDRAMSystem::PrintStats() {
    // Finish epoch output, remove last comma and append ]
    std::ofstream epoch_out(config_.json_epoch_name, std::ios_base::in |
                                                         std::ios_base::out |
                                                         std::ios_base::ate);
    epoch_out.seekp(-2, std::ios_base::cur);
    epoch_out.write("]", 1);
    epoch_out.close();

    std::ofstream json_out(config_.json_stats_name, std::ofstream::out);
    json_out << "{";

    // close it now so that each channel can handle it
    json_out.close();
    for (size_t i = 0; i < ctrls_.size(); i++) {
        ctrls_[i]->PrintFinalStats();
        if (i != ctrls_.size() - 1) {
            std::ofstream chan_out(config_.json_stats_name, std::ofstream::app);
            chan_out << "," << std::endl;
        }
    }
    json_out.open(config_.json_stats_name, std::ofstream::app);
    json_out << "}";

#ifdef THERMAL
    thermal_calc_.PrintFinalPT(clk_);
#endif  // THERMAL
}

void BaseDRAMSystem::ResetStats() {
    for (size_t i = 0; i < ctrls_.size(); i++) {
        ctrls_[i]->ResetStats();
    }
}

void BaseDRAMSystem::RegisterCallbacks(
    std::function<void(uint64_t)> read_callback,
    std::function<void(uint64_t)> write_callback) {
    // TODO this should be propagated to controllers
    read_callback_ = read_callback;
    write_callback_ = write_callback;
}

JedecDRAMSystem::JedecDRAMSystem(Config &config, const std::string &output_dir,
                                 std::function<void(uint64_t)> read_callback,
                                 std::function<void(uint64_t)> write_callback)
    : BaseDRAMSystem(config, output_dir, read_callback, write_callback) {
    if (config_.tile_instructions) {
        instruction_buffer_.reset(new PimInstructionBuffer(config_.instruction_capacity));
        instruction_sequencer_.reset(new PimInstructionSequencer(config_));
    }
    if (config_.IsHMC()) {
        std::cerr << "Initialized a memory system with an HMC config file!"
                  << std::endl;
        AbruptExit(__FILE__, __LINE__);
    }

    ctrls_.reserve(config_.channels);
    for (auto i = 0; i < config_.channels; i++) {
#ifdef THERMAL
        ctrls_.push_back(new Controller(i, config_, timing_, thermal_calc_));
#else
        ctrls_.push_back(new Controller(i, config_, timing_));
#endif  // THERMAL
    }

    if (config_.row_lockstep)
        row_scheduler_.reset(new PimRowScheduler(config_, ctrls_));
    int banks =  config_.ranks * config_.bankgroups * config_.banks_per_group;
    // std::cout<<"rank: "<<config_.ranks<<" bgs: "<<config_.bankgroups<<" bpg: "<<config_.banks_per_group<<std::endl;
    for (auto i = 0; i < config_.channels; i++) {
        auto chan_occupancy =
            std::vector<bool>(banks, false);
        bank_occupancy_.push_back(chan_occupancy);
    }
}

JedecDRAMSystem::~JedecDRAMSystem() {
    for (auto it = ctrls_.begin(); it != ctrls_.end(); it++) {
        delete (*it);
    }
}


bool JedecDRAMSystem::WillAcceptTransaction() const {
    if (instruction_buffer_) return instruction_buffer_->CanAcceptWord();
    return pim_trans_queue_.size() < pim_trans_queue_depth_;
}

bool JedecDRAMSystem::AddTransaction(uint64_t hex_addr) {
    if (instruction_buffer_) {
        bool accepted=instruction_buffer_->PushWord(hex_addr);
        if (accepted) last_req_clk_=clk_;
        return accepted;
    }
// Record trace - Record address trace for debugging or other purposes
#ifdef ADDR_TRACE
    address_trace_ << std::hex << hex_addr << std::dec << " "
                   << "PIM " << clk_ << std::endl;
#endif

    bool ok = WillAcceptTransaction();

    assert(ok);
    if (ok) {
        Transaction trans = Transaction(hex_addr);
        pim_trans_queue_.push_back(trans);

    }
    last_req_clk_ = clk_;
    return ok;
}

bool JedecDRAMSystem::WillAcceptTransaction(uint64_t hex_addr,
                                            bool is_write) const {
    int channel = GetChannel(hex_addr);
    return ctrls_[channel]->WillAcceptTransaction(hex_addr, is_write);
}

bool JedecDRAMSystem::AddTransaction(uint64_t hex_addr, bool is_write) {
// Record trace - Record address trace for debugging or other purposes
#ifdef ADDR_TRACE
    address_trace_ << std::hex << hex_addr << std::dec << " "
                   << (is_write ? "WRITE " : "READ ") << clk_ << std::endl;
#endif

    int channel = GetChannel(hex_addr);
    bool ok = ctrls_[channel]->WillAcceptTransaction(hex_addr, is_write);

    assert(ok);
    if (ok) {
        Transaction trans = Transaction(hex_addr, is_write);
        ctrls_[channel]->AddTransaction(trans);
    }
    last_req_clk_ = clk_;
    return ok;
}

void JedecDRAMSystem::ClockTick() {
    ReturnCompletedTransactions();
    if (row_scheduler_) row_scheduler_->Tick(clk_);
    bool wait_refresh = CheckRefreshWindow();
    if (instruction_buffer_) {
        instruction_buffer_->Tick(clk_);
        PimTickFeedback feedback;
        feedback.wait_refresh=wait_refresh;
        feedback.in_refresh=PimCommandsBlockedByRefresh();
        feedback.weight_pending=PendingPimSource(PimSource::WEIGHT);
        feedback.input_pending=PendingPimSource(PimSource::INPUT);
        feedback.output_pending=PendingPimSource(PimSource::OUTPUT);
        if (row_scheduler_) feedback.row_vectors_completed=row_scheduler_->CompletedVectors();
        auto result=instruction_sequencer_->Tick(clk_,*instruction_buffer_,feedback);
        execution_=instruction_sequencer_->Snapshot();
        if (execution_.in_pim) for (auto* controller:ctrls_) controller->in_pim=true;
        DispatchCommands(result.batch);
        if (result.row.vectors) row_scheduler_->Submit(std::move(result.row), clk_);
        turn_off=instruction_sequencer_->Complete();
    } else {
        ProcessPimTransaction();
        bool is_in_ref = PimCommandsBlockedByRefresh();
        SchedulePimCommands(wait_refresh, is_in_ref);
    }
    TickControllers();
    clk_++;
    if (clk_ % config_.epoch_period == 0) {
        PrintEpochStats();
    }
}

void JedecDRAMSystem::ReturnCompletedTransactions() {
    for (size_t i = 0; i < ctrls_.size(); i++) {
        // look ahead and return earlier
        while (true) {
            auto pair = ctrls_[i]->ReturnDoneTrans(clk_);
            if (pair.second == 1) {
                write_callback_(pair.first);
            } else if (pair.second == 0) {
                read_callback_(pair.first);
            } else {
                break;
            }
        }
    }

}

bool JedecDRAMSystem::CheckRefreshWindow() {
    // std::cout<<"Clock Cycle "<<clk_<<std::endl;

    // We calculate the refresh timing and pause PIM operations if a refresh is expected to occur during PIM operations.
    // lookup refresh_ in each controller to check refresh countdown
    // if countdown is lower than pim delay, pause issuing pim commands until refresh is done over all ranks
    bool wait_refresh = false;
    for (size_t i=0; i<ctrls_.size(); i++) {
        if (ctrls_[i]->pim_refresh_coming() &&
            (pim_config_.configured || (instruction_sequencer_ && execution_.in_pim))) {
            wait_refresh = true;
            // std::cout<<clk_ << "\tWait Refresh\n";
        }
    }


    return wait_refresh;
}

void JedecDRAMSystem::ProcessPimTransaction() {
    //*** Custom Transaction Queue Manager ***//
    // Pop a PIM transaction if the queue is not empty
    if (!pim_trans_queue_.empty()) {
        auto it = pim_trans_queue_.begin();
        DecodedPimTransaction decoded =
            PimTransactionDecoder::Decode(pim_trans_queue_.front().addr);

        switch (decoded.kind) {
        case PimTransactionKind::START_COMPUTATION: {
            if (pim_config_.configured && execution_.m != 0 &&
                execution_.n != 0 && execution_.k != 0) {
                execution_.in_pim = true;
                tile_executor_.reset(new PimTileExecutor(
                    config_, PimKernelContext::Capture(pim_config_, execution_)));
                tile_scheduler_.reset(new LegacyTileScheduler(
                    PimKernelContext::Capture(pim_config_, execution_), config_.channels));
                pim_trans_queue_.erase(it);
            }
            for (size_t i=0; i<ctrls_.size(); i++) {
                ctrls_[i]->in_pim = true;
            }

            break;
        }
        case PimTransactionKind::LOAD_DATAFLOW_CONFIG: {
            pim_config_.Load(decoded);
            execution_ = PimExecutionState();
            tile_executor_.reset();
            tile_scheduler_.reset();

            pim_trans_queue_.erase(it);
            break;
        }
        case PimTransactionKind::LOAD_WORKLOAD_CONFIG: {
            if (!pim_config_.configured) break;
            PimExecutionState& state = execution_;
            switch(decoded.load_type) {
                case 0: // M, weight
                    state.base_row_weight = decoded.base_row;
                    state.m = decoded.dim_value;
                    break;
                case 1: // K, output
                    state.base_row_output = decoded.base_row;
                    // std::cout<<base_row<<std::endl;
                    state.k = decoded.dim_value;
                    break;
                case 2: // N, input
                    state.base_row_in = decoded.base_row;
                    state.n = decoded.dim_value;
                    break;
                default:
                    std::cerr << "Invalid load type!"
                              << std::endl;
                    AbruptExit(__FILE__, __LINE__);
                    break;
            }
            pim_trans_queue_.erase(it);
            break;
        }
        }
    }

}

bool JedecDRAMSystem::PimCommandsBlockedByRefresh() const {
    bool is_in_ref = false;
    for (int i=0; i<ctrls_.size(); i++) {
        if (ctrls_[i]->IsInRef() || ctrls_[i]->pim_refresh_coming2())
            is_in_ref = true;
    }

    return is_in_ref;
}

bool JedecDRAMSystem::PendingPimSource(PimSource source) const {
    for (const auto* controller : ctrls_)
        if (controller->HasPendingPim(source)) return true;
    return false;
}

void JedecDRAMSystem::SchedulePimCommands(bool wait_refresh, bool is_in_ref) {
    if (!pim_config_.configured || !execution_.in_pim || is_in_ref) return;
    PimTickFeedback feedback;
    feedback.wait_refresh = wait_refresh;
    feedback.in_refresh = is_in_ref;
    feedback.weight_pending = PendingPimSource(PimSource::WEIGHT);
    feedback.input_pending = PendingPimSource(PimSource::INPUT);
    feedback.output_pending = PendingPimSource(PimSource::OUTPUT);
    PimDescriptorSource source;
    source.next_compute = [this](PimComputeBlock& block) {
        return tile_scheduler_->NextCompute(block);
    };
    source.next_output = [this](PimOutputBlock& block) {
        return tile_scheduler_->NextOutput(block);
    };
    auto result = tile_executor_->Tick(clk_, feedback, source);
    execution_ = tile_executor_->Snapshot();
    tile_scheduler_->ProjectProgress(execution_, result.compute_exhausted, result.output_exhausted);
    if (result.computation_finished)
        std::cout << clk_ << " End of Computation 0" << std::endl;
    if (result.finished)
        std::cout << clk_ << " Output Exhausted: Array0. Turn off PIM mode.\n";
    DispatchCommands(result.batch);
    if (result.finished) turn_off = true;
}

void JedecDRAMSystem::DispatchCommands(const PimCommandBatch& batch) {
    for (const auto& item : batch.operations) {
        const auto& operation = item.second;
        ctrls_[operation.commands.front().Channel()]->EnqueuePimOperation(item.first,operation);
    }
}

void JedecDRAMSystem::PrintStats() {
    BaseDRAMSystem::PrintStats();
    if (!instruction_buffer_) return;
    const auto& ingress=instruction_buffer_->Stats();
    const auto& sequence=instruction_sequencer_->Stats();
    nlohmann::json stats;
    stats["admitted_words"]=ingress.admitted_words;
    stats["decoded_instructions"]=ingress.decoded_instructions;
    stats["raw_high_watermark"]=ingress.raw_high_watermark;
    stats["slot_high_watermark"]=ingress.slot_high_watermark;
    stats["partial_packet_cycles"]=ingress.partial_packet_cycles;
    stats["raw_words"]=instruction_buffer_->RawSize();
    stats["instruction_slots"]=instruction_buffer_->SlotSize();
    stats["partial_packet"]=instruction_buffer_->HasPartialPacket();
    stats["dispatched_instructions"]=sequence.dispatched_instructions;
    stats["retired_instructions"]=sequence.retired_instructions;
    stats["compute_resource_stall_cycles"]=sequence.compute_resource_stall_cycles;
    stats["load_resource_stall_cycles"]=sequence.load_resource_stall_cycles;
    stats["output_resource_stall_cycles"]=sequence.output_resource_stall_cycles;
    stats["instruction_starvation_cycles"]=sequence.instruction_starvation_cycles;
    stats["backend_drained"]=sequence.backend_drained;
    stats["backend_drain_cycle"]=sequence.backend_drain_cycle;
    stats["end_completed"]=sequence.end_completed;
    stats["end_cycle"]=sequence.end_cycle;
    stats["cpu_backpressure_cycles"]=cpu_backpressure_cycles_;
    stats["status"]=sequence.end_completed ? "complete" : "incomplete";
    stats["executor_stage"]=execution_.iw_status;
    stats["simulated_cycles"]=clk_;
    stats["row_lockstep"] = bool(row_scheduler_);
    if (row_scheduler_) {
        const auto& rows = row_scheduler_->Stats();
        stats["row_operations_launched"] = rows.launched;
        stats["row_operations_completed"] = rows.completed;
        stats["row_boundary_wait_cycles"] = rows.boundary_wait_cycles;
    }
    std::ofstream out(config_.output_dir+"hbnpu_instruction.json");
    out << stats.dump(2) << '\n';
    if (!sequence.end_completed) std::cout << "HB-NPU instruction program incomplete at cycle " << clk_ << '\n';
}

void JedecDRAMSystem::TickControllers() {
    for (size_t i = 0; i < ctrls_.size(); i++) {
        ctrls_[i]->ClockTick();
    }

}

Command JedecDRAMSystem::GetReadyCommandPIM(Transaction trans, CommandType type) {
    bool first = true;
    bool sameornot = false;
    Command ready_cmd = Command();
    if (trans.targetChans.empty() || trans.targetBanks.empty()) std::cout<<"empty target Chans, Banks\t"<<trans<<'\n';
    for (auto& itC : trans.targetChans) {
        // We do not need to add more loops here since only these two dimensions are orthogonal to each other
        for (auto& itB : trans.targetBanks) {
            // TODO how to set rank, bankgroup, and column address (device width)?
            Address addr = Address((int) itC, 0, 0, (int) itB, (int) trans.row_addr,  (int) trans.col_num);
            Command cmd = Command(type, addr, trans.addr);
            std::cout<<clk_<<"\tWant To issue: " << cmd << '\n';
            ready_cmd = ctrls_[itC]->GetReadyCommand(cmd, clk_);
            if (first) {
                first = false;
                sameornot = cmd.cmd_type == ready_cmd.cmd_type;
            }
            else {
                if (sameornot != (cmd.cmd_type == ready_cmd.cmd_type)) {
                    std::cout <<clk_<< "\tSame or not fail:\n" << cmd << '\n' << ready_cmd << '\n';
                    return Command();
                }
            }
            if (!ready_cmd.IsValid() || bank_occupancy_[itC][itB]) {
                std::cout << clk_<<"\tGetReadyCommandPIM fail(" << ready_cmd << ") - Bank occupancy " << itC << itB << ": " << bank_occupancy_[itC][itB] << '\n';
                return Command();
            }
        }
    }
    return ready_cmd;
}



IdealDRAMSystem::IdealDRAMSystem(Config &config, const std::string &output_dir,
                                 std::function<void(uint64_t)> read_callback,
                                 std::function<void(uint64_t)> write_callback)
    : BaseDRAMSystem(config, output_dir, read_callback, write_callback),
      latency_(config_.ideal_memory_latency) {}

IdealDRAMSystem::~IdealDRAMSystem() {}

bool IdealDRAMSystem::AddTransaction(uint64_t hex_addr, bool is_write) {
    auto trans = Transaction(hex_addr, is_write);
    trans.added_cycle = clk_;
    infinite_buffer_q_.push_back(trans);
    return true;
}

void IdealDRAMSystem::ClockTick() {
    for (auto trans_it = infinite_buffer_q_.begin();
         trans_it != infinite_buffer_q_.end();) {
        if (clk_ - trans_it->added_cycle >= static_cast<uint64_t>(latency_)) {
            if (trans_it->is_write) {
                write_callback_(trans_it->addr);
            } else {
                read_callback_(trans_it->addr);
            }
            trans_it = infinite_buffer_q_.erase(trans_it++);
        }
        if (trans_it != infinite_buffer_q_.end()) {
            ++trans_it;
        }
    }

    clk_++;
    return;
}

}  // namespace dramsim3
