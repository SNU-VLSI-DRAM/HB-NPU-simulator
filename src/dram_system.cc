#include "dram_system.h"
#include "pim_transaction.h"
#include <assert.h>
#include <cmath>
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
    return pim_trans_queue_.size() < pim_trans_queue_depth_;
}

bool JedecDRAMSystem::AddTransaction(uint64_t hex_addr) {
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
    bool wait_refresh = CheckRefreshWindow();
    ProcessPimTransaction();
    bool is_in_ref = PimCommandsBlockedByRefresh();
    SchedulePimCommands(wait_refresh, is_in_ref);
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
        if (ctrls_[i]->pim_refresh_coming() && pim_config_.configured) {
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
    PimExecutionState& state = execution_;
    const int N_tile_size = 128;
    const int K_tile_size = std::min(config_.channels * 16, state.k);
    const int array_period = pim_config_.df == 0 ? config_.tCCD_S : config_.tCCD_L;
    const int weight_banks_reduce = pim_config_.df == 0 ? 8 : 16;
    const int N_tile_size_per_bank = std::min(state.n,
        (N_tile_size-1)/(config_.banks/weight_banks_reduce)+1);
    const bool output_ready = state.iw_status == 3;
    PimCommandBatch batch;
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
            if (state.weight_pending && !PendingPimSource(PimSource::WEIGHT)) {
                state.weight_pending = false;
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
            if (state.input_pending && !PendingPimSource(PimSource::INPUT)) {
                state.input_pending = false;
                int M_tile_it = state.m_it/pim_config_.m_tile_size;
                int N_tile_it = state.n_it/N_tile_size;
                // Retain the approved array geometry and activation overlap.
                // Trigger these counters on completed data vectors, never PRE.
                if ((state.k_tile_it+1)*K_tile_size >= state.k &&
                    state.m_it%pim_config_.m_tile_size == 0)
                    state.out_cnt = std::max(1,array_period*(3+16)-config_.tRCDWR);
                ++state.m_it;
                if (state.m_it%pim_config_.m_tile_size == 0 || state.m_it == state.m) {
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

    if (state.out_cnt == 0) ++state.output_valid;
    if (state.out_cnt != -1) --state.out_cnt;

    if (state.output_valid > 0 && output_ready) {
        const int M_tile_size_out = pim_config_.df == 1 ?
            (pim_config_.m_tile_size/128)*pim_config_.mcf : pim_config_.m_tile_size;
        const int M_out = pim_config_.df == 1 ? std::max(1,state.m*pim_config_.mcf/128) : state.m;
        const int N_out = pim_config_.df == 1 ? 128 : state.n;
        const int N_tile_size_out = 128;
        const int N_tile_num = (state.n-1)/N_tile_size_out+1;
        if (state.output_pending && !PendingPimSource(PimSource::OUTPUT)) {
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
                        assert(!PendingPimSource(PimSource::INPUT) && !PendingPimSource(PimSource::WEIGHT));
                        std::cout << clk_ << " Output Exhausted: Array0. Turn off PIM mode.\n";
                        state.in_pim = false;
                        turn_off = true;
                    }
                }
                --state.output_valid;
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
    DispatchCommands(batch);
}

void JedecDRAMSystem::DispatchCommands(const PimCommandBatch& batch) {
    for (const auto& item : batch.operations) {
        const auto& operation = item.second;
        ctrls_[operation.commands.front().Channel()]->EnqueuePimOperation(item.first,operation);
    }
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
