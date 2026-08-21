#ifndef __DRAM_SYSTEM_H
#define __DRAM_SYSTEM_H

#include <fstream>
#include <string>
#include <vector>

#include "common.h"
#include "configuration.h"
#include "controller.h"
#include "timing.h"

#ifdef THERMAL
#include "thermal.h"
#endif  // THERMAL

namespace dramsim3 {

class BaseDRAMSystem {
   public:
    BaseDRAMSystem(Config &config, const std::string &output_dir,
                   std::function<void(uint64_t)> read_callback,
                   std::function<void(uint64_t)> write_callback);
    virtual ~BaseDRAMSystem() {}
    void RegisterCallbacks(std::function<void(uint64_t)> read_callback,
                           std::function<void(uint64_t)> write_callback);
    void PrintEpochStats();
    void PrintStats();
    void ResetStats();

    virtual bool WillAcceptTransaction() const = 0;
    virtual bool AddTransaction(uint64_t hex_addr) = 0;
    virtual bool WillAcceptTransaction(uint64_t hex_addr,
                                       bool is_write) const = 0;
    virtual bool AddTransaction(uint64_t hex_addr, bool is_write) = 0;
    virtual void ClockTick() = 0;
    int GetChannel(uint64_t hex_addr) const;

    std::function<void(uint64_t req_id)> read_callback_, write_callback_;
    static int total_channels_;
    bool turn_off = false;

   protected:
    uint64_t id_;
    uint64_t last_req_clk_;
    Config &config_;
    Timing timing_;
    uint64_t parallel_cycles_;
    uint64_t serial_cycles_;


#ifdef THERMAL
    ThermalCalculator thermal_calc_;
#endif  // THERMAL

    uint64_t clk_;
    std::vector<Controller*> ctrls_;

#ifdef ADDR_TRACE
    std::ofstream address_trace_;
#endif  // ADDR_TRACE
};

// hmmm not sure this is the best naming...
class JedecDRAMSystem : public BaseDRAMSystem {
   public:
    JedecDRAMSystem(Config &config, const std::string &output_dir,
                    std::function<void(uint64_t)> read_callback,
                    std::function<void(uint64_t)> write_callback);
    ~JedecDRAMSystem();
    bool WillAcceptTransaction(uint64_t hex_addr, bool is_write) const override;
    bool WillAcceptTransaction() const override;
    bool AddTransaction(uint64_t hex_addr) override;
    bool AddTransaction(uint64_t hex_addr, bool is_write) override;
    void ClockTick() override;
    Command GetReadyCommandPIM(Transaction trans, CommandType type);

   private:
    void ReturnCompletedTransactions();
    bool CheckRefreshWindow();
    void ProcessPimTransaction();
    bool PimCommandsBlockedByRefresh() const;
    void SchedulePimCommands(bool wait_refresh, bool is_in_ref);
    void DispatchCommands(
        const std::vector<std::vector<Command>>& weight_commands,
        const std::vector<std::vector<Command>>& input_commands,
        const std::vector<std::vector<Command>>& output_commands);
    void TickControllers();

   public:
    // dataflow configuration
    int vcuts = -1;
    int hcuts = -1;
    int mcf = 1;
    int ucf = 1;
    int mc = 1;
    int df = -1;
    int vcuts_next = -1;
    int hcuts_next = -1;
    // TODO now it is same with all cuts, but it should be not
    int M_tile_size = 0;
    // TODO stride and kernel size also must be vectors
    int stride = 0;
    int kernel_size = 0;
    // workload configuration
    std::vector<uint64_t> base_rows_in;
    std::vector<uint64_t> base_rows_w;
    std::vector<uint64_t> base_rows_out;
    std::vector<int> M;
    std::vector<int> N;
    std::vector<int> K;
    // BLAS scheduler status
    std::vector<int> M_it;
    std::vector<int> N_it;
    std::vector<int> K_tile_it;
    std::vector<int> M_out_it;
    std::vector<int> N_out_tile_it;
    std::vector<bool> in_pim;
    std::vector<int> iw_status;
    std::vector<bool> in_act_placed;
    std::vector<bool> w_act_placed;
    std::vector<bool> out_act_placed;
    // NPU status
    std::vector<int> output_valid;
    std::vector<int> in_cnt;
    std::vector<int> out_cnt;
    std::vector<int> vpu_cnt;


    std::vector<std::vector<bool>> bank_occupancy_;
    std::vector<Transaction> pim_trans_queue_;
    uint64_t pim_trans_queue_depth_ = 32; //TODO
};

// Model a memorysystem with an infinite bandwidth and a fixed latency (possibly
// zero) To establish a baseline for what a 'good' memory standard can and
// cannot do for a given application
class IdealDRAMSystem : public BaseDRAMSystem {
   public:
    IdealDRAMSystem(Config &config, const std::string &output_dir,
                    std::function<void(uint64_t)> read_callback,
                    std::function<void(uint64_t)> write_callback);
    ~IdealDRAMSystem();
    bool WillAcceptTransaction(uint64_t hex_addr,
                               bool is_write) const override {
        return true;
    };
    bool AddTransaction(uint64_t hex_addr, bool is_write) override;
    bool WillAcceptTransaction() const override { return true;};
    bool AddTransaction(uint64_t hex_addr) override { return true;};
    void ClockTick() override;

   private:
    int latency_;
    std::vector<Transaction> infinite_buffer_q_;
};

}  // namespace dramsim3
#endif  // __DRAM_SYSTEM_H
