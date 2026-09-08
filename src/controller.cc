#include "controller.h"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <limits>

namespace dramsim3 {
namespace {
bool IsGlobalRead(const Command& cmd) {
    return cmd.cmd_type == CommandType::GH_READ ||
           cmd.cmd_type == CommandType::GH_READ_PRECHARGE;
}
}  // namespace

#ifdef THERMAL
Controller::Controller(int channel, const Config &config, const Timing &timing,
                       ThermalCalculator &thermal_calc)
#else
Controller::Controller(int channel, const Config &config, const Timing &timing)
#endif  // THERMAL
    : channel_id_(channel),
      clk_(0),
      config_(config),
      simple_stats_(config_, channel_id_),
      channel_state_(config, timing),
      cmd_queue_(channel_id_, config, channel_state_, simple_stats_),
      refresh_(config, channel_state_),
#ifdef THERMAL
      thermal_calc_(thermal_calc),
#endif  // THERMAL
      is_unified_queue_(config.unified_queue),
      row_buf_policy_(config.row_buf_policy == "CLOSE_PAGE"
                          ? RowBufPolicy::CLOSE_PAGE
                          : RowBufPolicy::OPEN_PAGE),
      last_trans_clk_(0),
      write_draining_(0) {
    if (is_unified_queue_) {
        unified_queue_.reserve(config_.trans_queue_size);
    } else {
        read_queue_.reserve(config_.trans_queue_size);
        write_buffer_.reserve(config_.trans_queue_size);
    }

#ifdef CMD_TRACE
    std::string trace_file_name = config_.output_prefix + "ch_" +
                                  std::to_string(channel_id_) + "cmd.trace";
    // std::cout << "Command Trace write to " << trace_file_name << std::endl;
    cmd_trace_.open(trace_file_name, std::ofstream::out);
    pim_trace_.open(config_.output_prefix + "ch_" +
                    std::to_string(channel_id_) + "pim.trace", std::ofstream::out);
#endif  // CMD_TRACE
}

std::pair<uint64_t, int> Controller::ReturnDoneTrans(uint64_t clk) {
    auto it = return_queue_.begin();
    while (it != return_queue_.end()) {
        if (clk >= it->complete_cycle) {
            if (it->is_write) {
                simple_stats_.Increment("num_writes_done");
            } else {
                simple_stats_.Increment("num_reads_done");
                simple_stats_.AddValue("read_latency", clk_ - it->added_cycle);
            }
            auto pair = std::make_pair(it->addr, it->is_write);
            it = return_queue_.erase(it);
            return pair;
        } else {
            ++it;
        }
    }
    return std::make_pair(-1, -1);
}

Command Controller::GetReadyCommand(Command& cmd, uint64_t clk) {
    Command ready = channel_state_.GetReadyCommand(cmd, clk);
    if ((IsGlobalRead(ready) || ready.cmd_type == CommandType::PIM_WRITE ||
         ready.cmd_type == CommandType::PIM_WRITE_PRECHARGE) && clk < next_global_io_cycle_)
        return Command();
    return ready;
}

Command Controller::GetReadyPimRead(Command& cmd, uint64_t clk, int bank_slot) {
    Command ready = GetReadyCommand(cmd, clk);
    if (!ready.IsValid() && IsGlobalRead(cmd) && bank_slot > 0) {
        // A bank pair is submitted together but consumes successive GH slots.
        // Probe the later slot so staggered same-bank timing does not insert
        // a bubble between pairs. Only reads may be queued using lookahead:
        // activation/precharge remain governed by current-cycle readiness.
        Command later = GetReadyCommand(
            cmd, clk + bank_slot * std::max(1, config_.tCCD_S));
        if (IsGlobalRead(later)) return later;
    }
    // The controller rechecks every queued command at its actual issue cycle.
    return ready;
}

bool Controller::pim_refresh_coming() {
    return refresh_.pim_refresh_coming();
}

void Controller::EnqueuePimOperation(PimSource source, const PimOperation& operation) {
    operation.Validate(config_);
    if (operation.commands.front().Channel() != channel_id_)
        throw std::invalid_argument("PIM operation submitted to wrong channel");
    auto& queue = source == PimSource::WEIGHT ? rd_w_cmds_ :
                  source == PimSource::INPUT ? rd_in_cmds_ : wr_cmds_;
    queue.push_back(operation);
    if (source == PimSource::INPUT) release_time.push_back(operation.release_cycle);
}

void Controller::EnqueuePimCommands(PimSource source, const std::vector<Command>& commands,
                                    const std::vector<int>& releases) {
    if (!releases.empty() && releases.size() != commands.size())
        throw std::invalid_argument("PIM release times must match commands");
    for (size_t i = 0; i < commands.size();) {
        PimOperation operation(commands[i], releases.empty() ? 0 : releases[i]);
        size_t count = operation.IsLocal() ? config_.banks : 1;
        if (i + count > commands.size()) throw std::invalid_argument("Incomplete LH broadcast");
        operation.commands.assign(commands.begin()+i,commands.begin()+i+count);
        for (size_t j=i; !releases.empty() && j<i+count; ++j)
            if (releases[j] != releases[i]) throw std::invalid_argument("Incompatible LH release times");
        EnqueuePimOperation(source,operation);
        i += count;
    }
}

void Controller::EnqueueWeightCommands(const std::vector<Command>& commands) {
    EnqueuePimCommands(PimSource::WEIGHT,commands,{});
}

void Controller::EnqueueInputCommands(const std::vector<Command>& commands,
                                      const std::vector<int>& releases) {
    if (commands.size() != releases.size())
        throw std::invalid_argument("PIM release times must match commands");
    EnqueuePimCommands(PimSource::INPUT,commands,releases);
}

void Controller::EnqueueOutputCommands(const std::vector<Command>& commands) {
    EnqueuePimCommands(PimSource::OUTPUT,commands,{});
}

bool Controller::HasPendingPim(PimSource source) const {
    return !(source == PimSource::WEIGHT ? rd_w_cmds_ :
             source == PimSource::INPUT ? rd_in_cmds_ : wr_cmds_).empty();
}

bool Controller::PimQueuesEmpty() const {
    return rd_w_cmds_.empty() && rd_in_cmds_.empty() && wr_cmds_.empty();
}

bool Controller::HasPendingPimInRank(int rank) const {
    for (const auto* queue : {&rd_w_cmds_, &rd_in_cmds_, &wr_cmds_}) {
        for (const auto& operation : *queue) {
            // Enqueue validation guarantees every target belongs to this rank.
            if (operation.commands.front().Rank() == rank) return true;
        }
    }
    return false;
}

bool Controller::SchedulePimQueue(std::vector<PimOperation>& queue, bool input) {
    // An unready target may not hold another bank's eligible transfer hostage.
    // Preserve per-bank order while permitting A,B,A,B interleaving.
    std::set<std::tuple<int,int,int>> blocked_banks;
    for (size_t i=0; i<queue.size(); ++i) {
        const auto& pending = queue[i];
        bool blocked = false;
        for (const auto& cmd : pending.commands)
            blocked |= blocked_banks.count(std::make_tuple(cmd.Rank(),cmd.Bankgroup(),cmd.Bank())) != 0;
        for (const auto& cmd : pending.commands)
            blocked_banks.emplace(cmd.Rank(),cmd.Bankgroup(),cmd.Bank());
        if (blocked) continue;
        auto ready = channel_state_.GetReadyPimOperation(pending,clk_);
        if (!ready.IsValid() || (ready.IsGlobal() && clk_ < next_global_io_cycle_) ||
            (ready.IsActivate() && channel_state_.IsRefreshWaiting())) continue;
#ifdef CMD_TRACE
        pim_trace_ << clk_ << " " << channel_id_ << " " << ready.Name() << " "
                   << ready.commands.front().Rank() << " ";
        for (size_t target=0; target<ready.commands.size(); ++target) {
            if (target) pim_trace_ << ",";
            pim_trace_ << ready.commands[target].Bankgroup() << ":" << ready.commands[target].Bank();
        }
        pim_trace_ << std::endl;
#endif
        // All bank prerequisites and aggregate activation cost were checked
        // without mutation. Issue the validated broadcast's physical effects.
        for (const auto& cmd : ready.commands) IssueCommand(cmd);
        if (ready.IsGlobal()) next_global_io_cycle_ = clk_ + std::max(1,config_.tCCD_S);
        if (ready.opcode == pending.opcode) {
            queue.erase(queue.begin()+i);
            if (input) release_time.erase(release_time.begin()+i);
        }
        return true;
    }
    return false;
}

void Controller::ClockTick() {
    // update refresh counter
    refresh_.ClockTick();

    bool cmd_issued = false;
    Command cmd;

    // priority 1: refresh command
    if (channel_state_.IsRefreshWaiting()) {
        cmd = cmd_queue_.FinishRefresh();
    }
    const bool refresh_command = cmd.IsValid();

    // cannot find a refresh related command or there's no refresh
    // priority 3: general command
    if (!cmd.IsValid() && PimQueuesEmpty()) {
        cmd = cmd_queue_.GetCommandToIssue();
    }

    // priority 2: pim command
    if (refresh_command || PimQueuesEmpty()) {

        if (cmd.IsValid()) {
            IssueCommand(cmd);
            cmd_issued = true;

            if (config_.enable_hbm_dual_cmd) {
                auto second_cmd = cmd_queue_.GetCommandToIssue();
                if (second_cmd.IsValid()) {
                    if (second_cmd.IsReadWrite() != cmd.IsReadWrite()) {
                        IssueCommand(second_cmd);
                        simple_stats_.Increment("hbm_dual_cmds");
                    }
                }
            }
        }
    }
    else {
        // TODO if second == 0, issue and set cmd_issue true. else, decrement by 1.
        cmd_issued = SchedulePimQueue(rd_w_cmds_, false) ||
                     SchedulePimQueue(rd_in_cmds_, true) ||
                     SchedulePimQueue(wr_cmds_, false);
        // rd_in_cmds_.clear(); //used in MT
    }


    // power updates pt 1
    for (int i = 0; i < config_.ranks; i++) {
        if (channel_state_.IsRankSelfRefreshing(i)) {
            simple_stats_.IncrementVec("sref_cycles", i);
        } else {
            bool all_idle = channel_state_.IsAllBankIdleInRank(i);
            if (all_idle) {
                simple_stats_.IncrementVec("all_bank_idle_cycles", i);
                channel_state_.rank_idle_cycles[i] += 1;
            } else {
                simple_stats_.IncrementVec("rank_active_cycles", i);
                // reset
                channel_state_.rank_idle_cycles[i] = 0;
            }
        }
    }

    // power updates pt 2: move idle ranks into self-refresh mode to save power
    if (config_.enable_self_refresh && !cmd_issued) {
        for (auto i = 0; i < config_.ranks; i++) {
            const bool rank_has_work = !cmd_queue_.rank_q_empty[i] ||
                                       HasPendingPimInRank(i);
            if (channel_state_.IsRankSelfRefreshing(i)) {
                // wake up!
                if (rank_has_work) {
                    auto addr = Address();
                    addr.rank = i;
                    auto cmd = Command(CommandType::SREF_EXIT, addr, -1);
                    cmd = channel_state_.GetReadyCommand(cmd, clk_);
                    if (cmd.IsValid()) {
                        IssueCommand(cmd);
                        break;
                    }
                }
            } else {
                if (!rank_has_work &&
                    channel_state_.rank_idle_cycles[i] >=
                        config_.sref_threshold) {
                    auto addr = Address();
                    addr.rank = i;
                    auto cmd = Command(CommandType::SREF_ENTER, addr, -1);
                    cmd = channel_state_.GetReadyCommand(cmd, clk_);
                    if (cmd.IsValid()) {
                        IssueCommand(cmd);
                        break;
                    }
                }
            }
        }
    }

    ScheduleTransaction();
    clk_++;
    cmd_queue_.ClockTick();
    simple_stats_.Increment("num_cycles");
    return;
}

bool Controller::WillAcceptTransaction(uint64_t hex_addr, bool is_write) const {
    if (is_unified_queue_) {
        return unified_queue_.size() < unified_queue_.capacity();
    } else if (!is_write) {
        return read_queue_.size() < read_queue_.capacity();
    } else {
        return write_buffer_.size() < write_buffer_.capacity();
    }
}

bool Controller::AddTransaction(Transaction trans) {
    trans.added_cycle = clk_;
    simple_stats_.AddValue("interarrival_latency", clk_ - last_trans_clk_);
    last_trans_clk_ = clk_;

    if (trans.is_write) {
        if (pending_wr_q_.count(trans.addr) == 0) {  // can not merge writes
            pending_wr_q_.insert(std::make_pair(trans.addr, trans));
            if (is_unified_queue_) {
                unified_queue_.push_back(trans);
            } else {
                write_buffer_.push_back(trans);
            }
        }
        trans.complete_cycle = clk_ + 1;
        return_queue_.push_back(trans);
        return true;
    } else {  // read
        // if in write buffer, use the write buffer value
        if (pending_wr_q_.count(trans.addr) > 0) {
            trans.complete_cycle = clk_ + 1;
            return_queue_.push_back(trans);
            return true;
        }
        pending_rd_q_.insert(std::make_pair(trans.addr, trans));
        if (pending_rd_q_.count(trans.addr) == 1) {
            if (is_unified_queue_) {
                unified_queue_.push_back(trans);
            } else {
                read_queue_.push_back(trans);
            }
        }
        return true;
    }
}


void Controller::ScheduleTransaction() {
    // determine whether to schedule read or write
    if (write_draining_ == 0 && !is_unified_queue_) {
        // we basically have a upper and lower threshold for write buffer
        if ((write_buffer_.size() >= write_buffer_.capacity()) ||
            (write_buffer_.size() > 8 && cmd_queue_.QueueEmpty())) {
            write_draining_ = write_buffer_.size();
        }
    }

    std::vector<Transaction> &queue =
        is_unified_queue_ ? unified_queue_
                          : write_draining_ > 0 ? write_buffer_ : read_queue_;
    for (auto it = queue.begin(); it != queue.end(); it++) {
        auto cmd = TransToCommand(*it);
        if (cmd_queue_.WillAcceptCommand(cmd.Rank(), cmd.Bankgroup(),
                                         cmd.Bank())) {
            if (!is_unified_queue_ && cmd.IsWrite()) {
                // Enforce R->W dependency
                if (pending_rd_q_.count(it->addr) > 0) {
                    write_draining_ = 0;
                    break;
                }
                write_draining_ -= 1;
            }
            cmd_queue_.AddCommand(cmd);
            queue.erase(it);
            break;
        }
    }
}

void Controller::IssueCommand(const Command &cmd) {
#ifdef CMD_TRACE
    cmd_trace_ << std::left << std::setw(18) << clk_ << " " << cmd << std::endl;
#endif  // CMD_TRACE
#ifdef THERMAL
    // add channel in, only needed by thermal module
    thermal_calc_.UpdateCMDPower(channel_id_, cmd, clk_);
#endif  // THERMAL
    // if read/write, update pending queue and return queue
    if (cmd.IsRead()) {
        auto num_reads = pending_rd_q_.count(cmd.hex_addr);
        if (in_pim == false) {
            if (num_reads == 0) {
                std::cerr << cmd.hex_addr << " not in read queue! " << std::endl;
                exit(1);
            }
        }
        // if there are multiple reads pending return them all
        while (num_reads > 0) {
            auto it = pending_rd_q_.find(cmd.hex_addr);
            it->second.complete_cycle = clk_ + config_.read_delay;
            return_queue_.push_back(it->second);
            pending_rd_q_.erase(it);
            num_reads -= 1;
        }
    } else if (cmd.IsWrite()) {
        // there should be only 1 write to the same location at a time
        auto it = pending_wr_q_.find(cmd.hex_addr);
        if (it == pending_wr_q_.end()) {
            std::cerr << cmd.hex_addr << " not in write queue!" << std::endl;
            exit(1);
        }

        auto wr_lat = clk_ - it->second.added_cycle + config_.write_delay;
        simple_stats_.AddValue("write_latency", wr_lat);
        pending_wr_q_.erase(it);
    }

    // must update stats before states (for row hits)

    UpdateCommandStats(cmd);
    channel_state_.UpdateTimingAndStates(cmd, clk_);
}

Command Controller::TransToCommand(const Transaction &trans) {
    auto addr = config_.AddressMapping(trans.addr);
    CommandType cmd_type;
    if (row_buf_policy_ == RowBufPolicy::OPEN_PAGE) {
        cmd_type = trans.is_write ? CommandType::WRITE : CommandType::READ;
    } else {
        cmd_type = trans.is_write ? CommandType::WRITE_PRECHARGE
                                  : CommandType::READ_PRECHARGE;
    }
    return Command(cmd_type, addr, trans.addr);
}

int Controller::QueueUsage() const { return cmd_queue_.QueueUsage(); }

void Controller::PrintEpochStats() {
    simple_stats_.Increment("epoch_num");
    simple_stats_.PrintEpochStats();
#ifdef THERMAL
    for (int r = 0; r < config_.ranks; r++) {
        double bg_energy = simple_stats_.RankBackgroundEnergy(r);
        thermal_calc_.UpdateBackgroundEnergy(channel_id_, r, bg_energy);
    }
#endif  // THERMAL
    return;
}

void Controller::PrintFinalStats() {
    simple_stats_.PrintFinalStats();

#ifdef THERMAL
    for (int r = 0; r < config_.ranks; r++) {
        double bg_energy = simple_stats_.RankBackgroundEnergy(r);
        thermal_calc_.UpdateBackgroundEnergy(channel_id_, r, bg_energy);
    }
#endif  // THERMAL
    return;
}

void Controller::UpdateCommandStats(const Command &cmd) {
    switch (cmd.cmd_type) {
        case CommandType::READ:
        case CommandType::READ_PRECHARGE:
            simple_stats_.Increment("num_read_cmds");
            if (channel_state_.RowHitCount(cmd.Rank(), cmd.Bankgroup(),
                                           cmd.Bank()) != 0) {
                simple_stats_.Increment("num_read_row_hits");
            }
            break;
        case CommandType::WRITE:
        case CommandType::WRITE_PRECHARGE:
            simple_stats_.Increment("num_write_cmds");
            if (channel_state_.RowHitCount(cmd.Rank(), cmd.Bankgroup(),
                                           cmd.Bank()) != 0) {
                simple_stats_.Increment("num_write_row_hits");
            }
            break;
        case CommandType::ACTIVATE:
            simple_stats_.Increment("num_act_cmds");
            break;
        case CommandType::LH_READ:
        case CommandType::LH_READ_PRECHARGE:
            simple_stats_.Increment("num_lh_read_cmds");
            if (channel_state_.RowHitCount(cmd.Rank(), cmd.Bankgroup(),
                                           cmd.Bank()) != 0) {
                simple_stats_.Increment("num_lh_read_row_hits");
            }
            break;
        case CommandType::GH_READ:
        case CommandType::GH_READ_PRECHARGE:
            simple_stats_.Increment("num_gh_read_cmds");
            if (channel_state_.RowHitCount(cmd.Rank(), cmd.Bankgroup(),
                                           cmd.Bank()) != 0) {
                simple_stats_.Increment("num_gh_read_row_hits");
            }
            break;
        case CommandType::PIM_WRITE:
        case CommandType::PIM_WRITE_PRECHARGE:
            simple_stats_.Increment("num_pim_write_cmds");
            if (channel_state_.RowHitCount(cmd.Rank(), cmd.Bankgroup(),
                                           cmd.Bank()) != 0) {
                simple_stats_.Increment("num_pim_write_row_hits");
            }
            break;
        case CommandType::PIM_ACTIVATE:
            simple_stats_.Increment("num_act_cmds");
            break;
        case CommandType::PRECHARGE:
            simple_stats_.Increment("num_pre_cmds");
            break;
        case CommandType::REFRESH:
            simple_stats_.Increment("num_ref_cmds");
            break;
        case CommandType::REFRESH_BANK:
            simple_stats_.Increment("num_refb_cmds");
            break;
        case CommandType::SREF_ENTER:
            simple_stats_.Increment("num_srefe_cmds");
            break;
        case CommandType::SREF_EXIT:
            simple_stats_.Increment("num_srefx_cmds");
            break;
        default:
            AbruptExit(__FILE__, __LINE__);
    }
}

}  // namespace dramsim3
