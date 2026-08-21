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
        if (ctrls_[i]->pim_refresh_coming() && pim_config_.vcuts != -1 && pim_config_.hcuts != -1) {
            wait_refresh = true;
            for (int j=0; j<pim_config_.vcuts*pim_config_.hcuts; j++) {
                partitions_[j].in_act_placed = false;
                partitions_[j].weight_act_placed = false;
                partitions_[j].output_act_placed = false;
            }
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
            int cuts = pim_config_.vcuts * pim_config_.hcuts;
            bool configured = true;
            for (int i=0; i<cuts; i++) {
                PimPartitionState& partition = partitions_[i];
                if ((decoded.launch_mask & (1 << i)) && (partition.m != 0 && partition.n != 0 && partition.k != 0));
                else {
                    configured = false;
                }
            }
            if (configured) {
                for (int i=0; i<cuts; i++) {
                    PimPartitionState& partition = partitions_[i];
                    if(decoded.launch_mask & (1 << i))
                        partition.in_pim = true;
                }
                pim_trans_queue_.erase(it);
            }
            for (size_t i=0; i<ctrls_.size(); i++) {
                ctrls_[i]->in_pim = true;
            }

            break;
        }
        case PimTransactionKind::LOAD_DATAFLOW_CONFIG: {
            partitions_.clear();

            pim_config_.LoadLayout(decoded);
            if (pim_config_.vcuts * pim_config_.hcuts > 1) // TODO
                for (int i=0; i<ctrls_.size(); i++) {
                    ctrls_[i]->wr_multitenant = true;
                }

            pim_config_.LoadTiling(decoded);

            int cuts = pim_config_.vcuts * pim_config_.hcuts;
            partitions_.assign(cuts, PimPartitionState());

            pim_trans_queue_.erase(it);
            break;
        }
        case PimTransactionKind::LOAD_WORKLOAD_CONFIG: {
            PimPartitionState& partition = partitions_[decoded.cut_no];
            switch(decoded.load_type) {
                case 0: // M, weight
                    partition.base_row_weight = decoded.base_row;
                    partition.m = decoded.dim_value;
                    break;
                case 1: // K, output
                    partition.base_row_output = decoded.base_row;
                    // std::cout<<base_row<<std::endl;
                    partition.k = decoded.dim_value;
                    break;
                case 2: // N, input
                    partition.base_row_in = decoded.base_row;
                    partition.n = decoded.dim_value;
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

void JedecDRAMSystem::SchedulePimCommands(bool wait_refresh, bool is_in_ref) {
    //*** HB-NPU Command Scheduler ***//
    // We are currently developing multi-tenant workload support in the NPU by partitioning the array and running them independently.
    // Please ignore these variables (~cut~) for now.
    int cuts = 0;
    if (pim_config_.vcuts != -1 && pim_config_.hcuts != -1) cuts = pim_config_.vcuts * pim_config_.hcuts;
    for (int i=0; i < cuts; i++) {
        PimPartitionState& partition = partitions_[i];
        if (!partition.in_pim || is_in_ref) continue;

        int vcut_no = i % pim_config_.vcuts;
        int cut_height = config_.channels / pim_config_.hcuts;
        int hcut_no = i / pim_config_.vcuts;
        int cut_width = config_.banks / pim_config_.vcuts;

        int N_tile_size = 128 / pim_config_.vcuts; // 128 : the number of PEs in a row
        int N_tile_it = partition.n_it / N_tile_size;
        int M_tile_it = partition.m_it / pim_config_.m_tile_size;
        int M_current_tile_size = partition.m < pim_config_.m_tile_size * (M_tile_it + 1) ? partition.m % pim_config_.m_tile_size : pim_config_.m_tile_size;
        int K_tile_size = std::min(cut_height * 16, partition.k); // 16: the number of PEs supported by a bank's io


        int weight_banks_reduce = pim_config_.df==0? 8:16; // Bank Interleaving option for weight loading
        std::vector<std::vector<Command>> in_cmds(cuts);
        std::vector<std::vector<Command>> w_cmds(cuts);

        bool output_ready = partition.iw_status == 3;

        // std::cout<<partition.iw_status<<"iw_status\n";
        // Our custom command scheduler changes iw_status value to switch the BLAS functions between loading data into PE array registers and streaming data into the array.
        // It manages matrix multiplication progress by monitoring and updating the BLAS status and NPU status
        switch (partition.iw_status) {
            case 0: { // data loading into PE registers
                CommandType act_type = CommandType::PIM_ACTIVATE;
                CommandType read_type = CommandType::GH_READ;
                CommandType readp_type = CommandType::GH_READ_PRECHARGE;


                int N_tile_size_per_bank = std::min(partition.n, (N_tile_size-1)/(cut_width/weight_banks_reduce) + 1);
                int col_offset = N_tile_it * (N_tile_size_per_bank * ((partition.k-1) / K_tile_size + 1)) + partition.k_tile_it * N_tile_size_per_bank + partition.n_it % N_tile_size; // n_it incremented by N_tile_size when n_it % N_tile_size_per_bank == 0 (but not with N_tile_size)
                // Scheduler generates the commands for a data vector, divided into multiple channels and sends them to command queues in the corresponding channel controllers.
                for (int j=0; j<cut_height; j++) {
                    // It can read multiple banks for bank interleaving or only one bank per channel
                    for (int k=0; k<cut_width/weight_banks_reduce; k++) {
                        int ch = hcut_no * cut_height + j;
                        int bk = vcut_no * cut_width + k * weight_banks_reduce;
                        int bg = bk / config_.banks_per_group;
                        bk = bk % config_.banks_per_group;
                        // building memory address by combining base physical address and BLAS configuration
                        Address addr = Address(ch, 0, bg, bk, partition.base_row_weight + col_offset/(config_.columns/config_.BL),  col_offset % (config_.columns/config_.BL));
                        uint64_t hex_addr = config_.AddressUnmapping(addr);
                        // generate read-precharge command if this is the last access to read the tile.
                        CommandType cmd_type;
                        bool exit = ((partition.n_it+1) % N_tile_size_per_bank == 0 && (N_tile_size == N_tile_size_per_bank || (partition.n_it+1) % N_tile_size != 0));
                        cmd_type = (addr.column + 1) % std::min(partition.n, 128 / config_.banks * weight_banks_reduce) == 0 || (addr.column + 1) % (config_.columns / config_.BL) == 0 || exit ? readp_type : read_type;
                        Command cmd = Command(cmd_type, addr, hex_addr);
                        Command ready_cmd = ctrls_[ch]->GetReadyCommand(cmd, clk_);
                        // If a command cannot be executed in some channels due to timing constraints, flush the commands going to other channels and try again later.
                        // This is to prevent the commands from being sent multiple times.
                        if (!ready_cmd.IsValid()) {
                            w_cmds[i].clear();
                            break;
                        }
                        else {
                            w_cmds[i].push_back(ready_cmd);
                            if (w_cmds[i].begin()->cmd_type != ready_cmd.cmd_type) {
                                w_cmds[i].clear();
                                break;
                            }
                        }
                    }
                    if (w_cmds[i].empty()) break;
                }


                if (w_cmds[i].empty()) break;
                // Check if the activation command was already sent.
                if (w_cmds[i].begin()->cmd_type == act_type) {
                    if (partition.weight_act_placed || wait_refresh) {
                        w_cmds[i].clear();
                        break;
                    }
                    else
                        partition.weight_act_placed = true;
                }
                //
                else {
                    if (w_cmds[i].begin()->cmd_type == readp_type) {
                        partition.weight_act_placed = false;
                    }
                    if (pim_config_.df == 1 && w_cmds[i].begin()->cmd_type == CommandType::PRECHARGE) {
                        break;
                    }

                    // increment BLAS iterators
                    partition.n_it++;
                    if (partition.n_it % N_tile_size_per_bank == 0 && (N_tile_size == N_tile_size_per_bank || partition.n_it % N_tile_size != 0)) {
                        partition.n_it = N_tile_size * N_tile_it;
                        partition.iw_status++;
                    }
                }


                break;
            }
            case 1: { // Finished data loading
                // wait npu signals
                // For non-multi-tenant cases, advance to next stage immediately.
                partition.iw_status++;
                partition.vpu_cnt = 1;
                if (cuts == 1) { //partition.n==1) { //TODO support for MT
                    for (int j=0; j<partitions_.size(); j++) {
                        if (partitions_[j].iw_status == 0 || partitions_[j].iw_status == 3) {
                            partition.iw_status--;
                            break;
                        }
                    }
                }
                break;
            }
            case 2: { // Streaming data into PE array
                CommandType act_type = CommandType::PIM_ACTIVATE;
                // Weight stationary dataflow opts to use Global HB, while TS-GEMM/GEMV dataflow uses Local HB.
                CommandType read_type = pim_config_.df == 0 ? CommandType::GH_READ : CommandType::LH_READ;
                CommandType readp_type = pim_config_.df == 0 ? CommandType::GH_READ_PRECHARGE : CommandType::LH_READ_PRECHARGE;
                partition.vpu_cnt--;
                partition.vpu_cnt = std::max(0, partition.vpu_cnt);

                int col_offset = M_tile_it * (pim_config_.m_tile_size * ((partition.k-1) / K_tile_size + 1)) + partition.k_tile_it * M_current_tile_size + partition.m_it % pim_config_.m_tile_size;
                bool mixed = false;
                Command mixed_cmd;
                // Scheduler generates the commands for a data vector, divided into multiple channels and sends them to command queues in the corresponding channel controllers.
                for (int j=0; j<cut_height; j++) {
                    // It can generate commands for multiple banks per channel simultaneously depending on the dataflow configuration.
                    for (int k=0; k<pim_config_.mc; k++) {

                        int ch = hcut_no * cut_height + j;
                        int bk = vcut_no * cut_width + k*(cut_width/pim_config_.mc);
                        if (pim_config_.df==0) bk++;
                        int bg = bk / config_.banks_per_group;
                        bk = bk % config_.banks_per_group;

                        // building memory address by combining base physical address and BLAS configuration
                        Address addr = Address(ch, 0, bg, bk, partition.base_row_in + col_offset/(config_.columns/config_.BL),  col_offset % (config_.columns/config_.BL));
                        uint64_t hex_addr = config_.AddressUnmapping(addr);
                        bool close = partition.m_it + 1 == partition.m; // prevent closing between tiles
                        bool close2 = (partition.k_tile_it+1) * K_tile_size >= partition.k; // leave open in WS since batch size is too small in LLMs
                        bool close3 = pim_config_.df==0?close2 && close:close;
                        // generate read-precharge command if this is the last access to read the tile.
                        CommandType cmd_type = close3 || addr.column == config_.columns / config_.BL - 1 ? readp_type : read_type;
                        Command cmd = Command(cmd_type, addr, hex_addr);
                        Command ready_cmd = ctrls_[ch]->GetReadyCommand(cmd, clk_);
                        // If a command cannot be executed in some channels due to timing constraints, flush the commands going to other channels and try again later.
                        // This is to prevent the commands from being sent multiple times.
                        if (!ready_cmd.IsValid()) {
                            in_cmds[i].clear();
                            break;
                        }
                        else {
                            in_cmds[i].push_back(ready_cmd);
                            if (in_cmds[i].begin()->cmd_type != ready_cmd.cmd_type) {
                                if (mixed) {
                                    if (mixed_cmd.cmd_type != in_cmds[i].begin()->cmd_type && mixed_cmd.cmd_type != ready_cmd.cmd_type) {
                                        std::cout<<"3 ops mixed: "<<mixed_cmd<<*in_cmds[i].begin()<<ready_cmd<<std::endl;
                                    }
                                }
                                else {
                                    mixed = true;
                                    mixed_cmd = ready_cmd;
                                }

                            }
                        }
                    }
                }
                if(cuts > 1 && in_cmds[i].size() != cut_height) {
                    in_cmds[i].clear();
                    break;
                }
                if (mixed) {
                    for (auto it = in_cmds[i].begin(); it != in_cmds[i].end();) {
                        if (it->cmd_type == read_type || it->cmd_type == readp_type) {
                            it = in_cmds[i].erase(it);
                        }
                        else
                            it++;
                    }
                }

                if (in_cmds[i].empty()) break;

                // Check if the activation command was already sent.
                if (in_cmds[i].begin()->cmd_type == act_type) {
                    if ((partition.in_act_placed) || wait_refresh) {
                        in_cmds[i].clear();
                        break;
                    }
                    else{
                        partition.in_act_placed = true;

                    }
                }
                else {

                    if (in_cmds[i].begin()->cmd_type == readp_type) {
                        partition.in_act_placed = false;
                    }
                    if (partition.vpu_cnt!=0){
                        in_cmds[i].clear();
                        break;
                    }

                    assert(pim_config_.m_tile_size > 128/pim_config_.vcuts);

                    // Update NPU status. Countdown the operation delay.
                    if ((partition.k_tile_it+1) * K_tile_size >= partition.k && partition.m_it % pim_config_.m_tile_size == 0) {
                        partition.out_cnt = std::max(1, config_.tCCD_L * (3 + 16) - config_.tRCDWR);
                    }


                    // Increment Iterators
                    partition.m_it++;
                    if (partition.m_it % pim_config_.m_tile_size == 0 || partition.m_it == partition.m) {
                        partition.in_cnt = std::max(1, config_.tCCD_L * std::max(128/(pim_config_.vcuts*pim_config_.mc), 16) - config_.tRCDRD);
                        partition.iw_status++;
                        partition.m_it = pim_config_.m_tile_size * M_tile_it;
                        partition.k_tile_it++;

                        if (partition.k_tile_it * K_tile_size >= partition.k) {
                            // partition.out_cnt = 3;
                            partition.k_tile_it = 0;
                            partition.n_it = N_tile_size * (N_tile_it+1);
                            if (partition.n_it >= partition.n) {
                                partition.n_it = 0;
                                partition.m_it = pim_config_.m_tile_size * (M_tile_it + 1);
                                if (partition.m_it >= partition.m) {
                                    std::cout<<clk_<<" End of Computation "<<i<<std::endl;
                                    partition.in_cnt = -1;
                                }
                            }
                        }
                    }
                }
                break;
            }
            case 3: {// Finished input
                // Lookup NPU status
                // Wait until PE array is available for loading a new tile.
                if (partition.in_cnt == -1) break;
                else {

                    partition.in_cnt = std::max(0, partition.in_cnt - 1);
                    if (partition.in_cnt == 0 && partition.output_valid == 0)
                        partition.iw_status = 0;
                    break;
                }
                break;
            }
            default: {
                break;
            }
        }



        // Update NPU status
        if (partition.out_cnt == 0) partition.output_valid++;
        if (partition.out_cnt != -1) partition.out_cnt--;


        std::vector<std::vector<Command>> out_cmds(cuts);


        // Writing Output from NPU to DRAM
        // Command Scheduler lookups the NPU status to check if the output data is ready to be sent to DRAM.
        bool out_enable = cut_height / pim_config_.vcuts > 0 || vcut_no % 2 == 0;
        if (partition.output_valid > 0 && output_ready && out_enable) {
            int vcut_out_no = partition.m == 1 ? vcut_no : pim_config_.vcuts == 16 ? vcut_no / 2 : (vcut_no + partition.n_out_tile_it) % pim_config_.vcuts; // relates to channel number
            int M_tile_size_out = pim_config_.df == 1 ? (pim_config_.m_tile_size/128)*pim_config_.mcf : pim_config_.m_tile_size;
            int M_out_tile_it = partition.m_out_it / M_tile_size_out;
            int M_out = pim_config_.df == 1 ? std::max(1, partition.m*pim_config_.mcf / 128) : partition.m;
            int M_out_current_tile_size = M_out < M_tile_size_out * (M_out_tile_it + 1) ? M_out % M_tile_size_out : M_tile_size_out;
            int N_out = pim_config_.df == 1 ? 128 : partition.n;
            int N_tile_size_out = pim_config_.df == 1 ? 128 : N_tile_size;
            int N_tile_num = (partition.n-1) / N_tile_size_out + 1;
            int N_tile_num_ch = (N_tile_num) / pim_config_.vcuts; // varies by channels to be accessed
            N_tile_num_ch += N_tile_num % pim_config_.vcuts > partition.n_out_tile_it % pim_config_.vcuts ? 1 : 0;
            int N_tile_it_ch = partition.n_out_tile_it / pim_config_.vcuts;
            int col_offset = M_out_tile_it * (M_tile_size_out * N_tile_num_ch) + N_tile_it_ch * M_out_current_tile_size + partition.m_out_it % M_tile_size_out;

            // Scheduler generates the commands for a data vector, divided into multiple channels and sends them to command queues in the corresponding channel controllers.
            int cut_height_out = cut_height < pim_config_.vcuts ? 1 : cut_height / pim_config_.vcuts;
            for (int j=0; j<cut_height_out; j++) {

                // It can generate commands for multiple banks per channel simultaneously depending for bank interleaving.
                int ch = hcut_no * cut_height + vcut_out_no * cut_height_out + j;
                int k_bound = pim_config_.df == 1 ? 1 : partition.m == 1 || true ? pim_config_.mc : 1;
                for (int k=0; k<k_bound; k++) {
                    int bk = vcut_no * cut_width + k*(cut_width/pim_config_.mc);
                    if (pim_config_.df != 1) bk++;
                    int bg = bk / config_.banks_per_group;
                    bk = bk % config_.banks_per_group;
                    if (pim_config_.df==0) bk += 2;
                    // building memory address by combining base physical address and BLAS configuration
                    Address addr = Address(ch, 0, bg, bk, partition.base_row_output + col_offset/(config_.columns/config_.BL),  col_offset % (config_.columns/config_.BL));
                    // Address addr = Address(ch, 0, bg, bk, partition.base_row_output + col_offset,  col_offset % (config_.columns/config_.BL));
                    uint64_t hex_addr = config_.AddressUnmapping(addr);
                    // generate write-precharge command if this is the last access to write the output tile.
                    bool close = partition.m_out_it + 1 == M_out;
                    CommandType cmd_type = close || addr.column == config_.columns / config_.BL - 1 ? CommandType::PIM_WRITE_PRECHARGE : CommandType::PIM_WRITE;
                    Command cmd = Command(cmd_type, addr, hex_addr);
                    Command ready_cmd = ctrls_[ch]->GetReadyCommand(cmd, clk_);

                    // If a command cannot be executed in some channels due to timing constraints, flush the commands going to other channels and try again later.
                    // This is to prevent the commands from being sent multiple times.
                    if (!ready_cmd.IsValid()) {
                        out_cmds[i].clear();
                        break;
                    }
                    else {
                        out_cmds[i].push_back(ready_cmd);
                        if (out_cmds[i].begin()->cmd_type != ready_cmd.cmd_type) {
                            out_cmds[i].clear();
                            break;
                        }
                    }
                }
                if (out_cmds[i].empty()) break;
            }

            // Check if the activation command was already sent.
            if (!out_cmds[i].empty()) {
                if (out_cmds[i].begin()->cmd_type == CommandType::PIM_ACTIVATE) {
                    if (partition.output_act_placed || wait_refresh) {
                        out_cmds[i].clear();
                    }
                    else {
                        partition.output_act_placed = true;
                    }
                }
                else {
                    if (out_cmds[i].begin()->cmd_type == CommandType::PIM_WRITE_PRECHARGE) {
                        partition.output_act_placed = false;
                    }

                    // Increment Iterators
                    partition.m_out_it++;
                    if (partition.m_out_it % M_tile_size_out == 0 || partition.m_out_it == M_out) {
                        partition.m_out_it = M_tile_size_out * M_out_tile_it;
                        partition.n_out_tile_it++;
                        if (partition.n_out_tile_it * N_tile_size_out >= N_out) {
                            partition.n_out_tile_it = 0;
                            partition.m_out_it = M_tile_size_out * (M_out_tile_it+1);
                            if (partition.m_out_it >= M_out) {
                                assert(partition.in_cnt == -1);
                                std::cout<<clk_<<" Output Exhausted: Array"<<i<<". Turn off PIM mode.\n";
                                partition.in_pim = false;
                                if (cut_height < pim_config_.vcuts) partitions_[i+1].in_pim = false;
                                turn_off = true;
                                for (size_t j = 0; j < partitions_.size(); j++) {
                                    if (partitions_[j].in_pim) {
                                        turn_off = false;
                                    }

                                }
                            }

                        }

                        partition.output_valid--;
                        if (cut_height < pim_config_.vcuts) partitions_[i+1].output_valid--;
                        // Output Tile Finished
                    }
                }
            }

        }

        // Finally the scheduler sends the aggregated commands to channel controllers by pushing them into custom command queues, which are managed in-order.
        DispatchCommands(w_cmds, in_cmds, out_cmds);

    }




}

void JedecDRAMSystem::DispatchCommands(
    const std::vector<std::vector<Command>>& weight_commands,
    const std::vector<std::vector<Command>>& input_commands,
    const std::vector<std::vector<Command>>& output_commands) {
    for (auto& it: weight_commands) {
        for (auto& it2: it) {
           // std::cout<<clk_<<" "<<it<<std::endl;
            ctrls_[it2.Channel()]->rd_w_cmds_.push_back(it2);
        }
    }
    for (auto& it: input_commands) {
        for (auto& it2: it) {
            ctrls_[it2.Channel()]->rd_in_cmds_.push_back(it2);
            int release_time_ = clk_;
            if (it2.cmd_type == CommandType::PIM_ACTIVATE) release_time_ += 0;  // + (it.Channel() % cut_height)*config_.tCCD_S);
            ctrls_[it2.Channel()]->release_time.push_back(release_time_);
        }
    }
    for (auto& it: output_commands) {
        for (auto& it2: it) {
            ctrls_[it2.Channel()]->wr_cmds_.push_back(it2);
        }
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
