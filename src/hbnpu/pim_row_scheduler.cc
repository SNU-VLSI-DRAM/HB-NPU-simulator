#include "hbnpu/pim_row_scheduler.h"

#include "json.hpp"
namespace dramsim3 {
PimRowScheduler::PimRowScheduler(const Config& hw,
                                 const std::vector<Controller*>& controllers)
    : controllers_(controllers), trace_(hw.output_dir + "hbnpu_rows.jsonl") {
    if (!trace_) throw std::runtime_error("Cannot open HB-NPU row trace");
}
void PimRowScheduler::Submit(PimRowOperation row, uint64_t cycle) {
    if (pending_ || row.channels.size() != controllers_.size() ||
        row.vectors <= 0)
        throw std::logic_error("Invalid or overlapping PIM row submission");
    for (auto* c : controllers_)
        if (row.restart_budget >= c->PimRefreshInterval())
            throw std::invalid_argument(
                "PIM row cannot fit post-refresh recovery/preparation window");
    row_ = std::move(row);
    for (size_t ch = 0; ch < controllers_.size(); ++ch)
        controllers_[ch]->PreparePimRow(row_.channels[ch],
                                        row_.operations_per_vector);
    completed_vectors_ = 0;
    requested_ = cycle;
    pending_ = true;
}
void PimRowScheduler::Tick(uint64_t cycle) {
    if (!pending_) return;
    if (!running_) {
        for (auto* c : controllers_) {
            if (!c->CanLaunchPimRow(row_.refresh_budget)) {
                ++stats_.boundary_wait_cycles;
                return;
            }
        }
        for (auto* c : controllers_) c->LaunchPimRow(cycle);
        running_ = true;
        stats_.last_start = cycle;
        ++stats_.launched;
        return;
    }
    size_t issued = controllers_.front()->PimRowIssued();
    for (auto* c : controllers_)
        if (c->PimRowIssued() != issued)
            throw std::runtime_error(
                "PIM row predictability violation: channel skew");
    completed_vectors_ = issued / row_.operations_per_vector;
    if (issued == row_.channels.front().size()) {
        stats_.last_end = cycle - 1;
        if (stats_.last_end !=
            stats_.last_start + row_.channels.front().back().release_cycle)
            throw std::runtime_error(
                "PIM row predictability violation: duration");
        nlohmann::json record;
        record["id"] = stats_.completed;
        record["request_cycle"] = requested_;
        record["start_cycle"] = stats_.last_start;
        record["last_issue_cycle"] = stats_.last_end;
        record["vectors"] = row_.vectors;
        record["operations_per_vector"] = row_.operations_per_vector;
        record["vector_period"] = row_.period;
        record["m_origin"] = row_.m_origin;
        record["k_tile"] = row_.k_tile;
        const auto& first = row_.channels.front().front().commands.front();
        record["row"] = first.Row();
        record["first_column"] = first.Column();
        trace_ << record.dump() << std::endl;
        for (auto* c : controllers_) c->FinishPimRow();
        ++stats_.completed;
        pending_ = running_ = false;
        row_ = PimRowOperation{};
    }
}
}  // namespace dramsim3
