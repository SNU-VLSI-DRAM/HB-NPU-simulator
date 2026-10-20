#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "catch.hpp"
#include "hbnpu/pim_row_scheduler.h"
#include "json.hpp"
using namespace dramsim3;
namespace {
PimKernelContext Context(bool local) {
    PimKernelContext c;
    c.m = 65;
    c.n = 128;
    c.k = 128;
    c.base_row_in = 100;
    c.dataflow.configured = true;
    c.dataflow.df = local ? 1 : 0;
    c.dataflow.mc = local ? 16 : 2;
    c.dataflow.mcf = 2;
    c.dataflow.ucf = local ? 8 : 1;
    c.dataflow.m_tile_size = 128;
    return c;
}

struct ExpectedRow {
    uint64_t start;
    int origin, vectors;
};

#ifdef CMD_TRACE
// Read the physical effects, independently of descriptor release offsets.
void CheckRowTrace(const Config& hw, bool local,
                   const std::vector<ExpectedRow>& rows) {
    struct Read {
        uint64_t cycle;
        int channel, rank, bg, bank, row, column;
    };
    for (int ch = 0; ch < hw.channels; ++ch) {
        std::ifstream file(hw.output_dir + "dramsim3ch_" +
                           std::to_string(ch) + "cmd.trace");
        REQUIRE(file.good());
        std::vector<Read> reads;
        Read read;
        std::string kind, row, column;
        while (file >> read.cycle >> kind >> read.channel >> read.rank >>
               read.bg >> read.bank >> row >> column) {
            if (kind.find("gh_read") != 0 && kind.find("lh_read") != 0)
                continue;
            REQUIRE(kind.find(local ? "lh_read" : "gh_read") == 0);
            read.row = std::stoi(row, nullptr, 16);
            read.column = std::stoi(column, nullptr, 16);
            reads.push_back(read);
        }
        REQUIRE(file.eof());
        size_t count = 0;
        for (const auto& expected : rows)
            count += expected.vectors * (local ? 16 : 2);
        REQUIRE(reads.size() == count);
        size_t index = 0;
        for (const auto& expected : rows) {
            for (int v = 0; v < expected.vectors; ++v) {
                for (int slot = 0; slot < (local ? 16 : 2); ++slot) {
                    const auto& actual = reads[index++];
                    const int bank = local ? slot : 1 + slot * 8;
                    REQUIRE(actual.cycle == expected.start + v * 2 +
                                                (local ? 0 : slot));
                    REQUIRE(actual.channel == ch);
                    REQUIRE(actual.rank == 0);
                    REQUIRE(actual.bg == bank / 4);
                    REQUIRE(actual.bank == bank % 4);
                    REQUIRE(actual.row == 100 + expected.origin / 32);
                    REQUIRE(actual.column == expected.origin % 32 + v);
                }
            }
        }
    }
}
#endif
}  // namespace

TEST_CASE("Row admission checks exact refresh slack", "[pim-row-boundary]") {
    // Catch relaxing <= to <, tick-order drift, or an unready bank being
    // mistaken for refresh rejection. All channels are ready before the probe.
    for (bool local : {false, true}) {
        for (bool tail : {false, true}) {
            for (int delta : {-1, 0, 1}) {
                CAPTURE(local, tail, delta);
                char temp[] = "/tmp/hbnpu-row-boundary-XXXXXX";
                REQUIRE(mkdtemp(temp) != nullptr);
                Config hw("configs/HBM2_8Gb_x128_pim.ini", temp);
                hw.tREFI = 1000;
                Timing timing(hw);
                std::vector<std::unique_ptr<Controller>> owned;
                std::vector<Controller*> controllers;
                for (int ch = 0; ch < hw.channels; ++ch) {
                    owned.emplace_back(new Controller(ch, hw, timing));
                    controllers.push_back(owned.back().get());
                    controllers.back()->in_pim = true;
                }
                PimRowScheduler scheduler(hw, controllers);
                const int origin = tail ? 64 : 0;
                const int vectors = tail ? 1 : 32;
                // Hand-derived for the supplied HBM timing: last beat + 65.
                const uint64_t budget = tail ? (local ? 65 : 66)
                                             : (local ? 127 : 128);
                auto row = PimRowOperation::Build(hw, Context(local), origin,
                                                  0, 65);
                REQUIRE(row.refresh_budget == budget);
                scheduler.Submit(std::move(row), 0);
                const int probe_cycle = 1000 - (int(budget) + delta);
                int cycle = 0;
                for (; cycle < probe_cycle; ++cycle)
                    for (auto* c : controllers) c->ClockTick();
                for (auto* c : controllers) {
                    REQUIRE(c->CanLaunchPimRow(0));
                    REQUIRE(c->PimRowIssued() == 0);
                    REQUIRE(c->CanLaunchPimRow(budget) == (delta == 1));
                }
                scheduler.Tick(cycle);
                REQUIRE(scheduler.Running() == (delta == 1));
                for (auto* c : controllers) {
                    c->ClockTick();
                    REQUIRE(c->PimRowIssued() == (delta == 1 ? 1 : 0));
                }
                ++cycle;
                for (; cycle < 3000 && scheduler.Stats().completed == 0;
                     ++cycle) {
                    scheduler.Tick(cycle);
                    for (auto* c : controllers) c->ClockTick();
                }
                REQUIRE(scheduler.Stats().completed == 1);
                REQUIRE(scheduler.CompletedVectors() == vectors);
                const auto start = scheduler.Stats().last_start;
                if (delta == 1) {
                    REQUIRE(start == uint64_t(probe_cycle));
                    REQUIRE(scheduler.Stats().last_end < 1000);
                } else {
                    REQUIRE(start >= 1260);
                    REQUIRE(scheduler.Stats().last_end < 2000);
                }
                REQUIRE(scheduler.Stats().last_end ==
                        start + (vectors - 1) * 2 + (local ? 0 : 1));
#ifdef CMD_TRACE
                CheckRowTrace(hw, local, {{start, origin, vectors}});
#endif
            }
        }
    }
}

TEST_CASE("Consecutive row segments reset progress", "[pim-row-boundary]") {
    // Catch stale completion leaking into a newly submitted segment, including
    // a one-vector tail immediately after two full physical rows.
    for (bool local : {false, true}) {
        CAPTURE(local);
        char temp[] = "/tmp/hbnpu-row-consecutive-XXXXXX";
        REQUIRE(mkdtemp(temp) != nullptr);
        Config hw("configs/HBM2_8Gb_x128_pim.ini", temp);
        Timing timing(hw);
        std::vector<std::unique_ptr<Controller>> owned;
        std::vector<Controller*> controllers;
        for (int ch = 0; ch < hw.channels; ++ch) {
            owned.emplace_back(new Controller(ch, hw, timing));
            controllers.push_back(owned.back().get());
            controllers.back()->in_pim = true;
        }
        PimRowScheduler scheduler(hw, controllers);
        std::vector<ExpectedRow> expected;
        int cycle = 0;
        uint64_t previous_end = 0;
        for (int segment = 0; segment < 3; ++segment) {
            CAPTURE(segment);
            const int origin = segment * 32;
            const int vectors = segment == 2 ? 1 : 32;
            auto row = PimRowOperation::Build(hw, Context(local), origin, 0, 65);
            REQUIRE(row.vectors == vectors);
            REQUIRE(row.channels[0][0].commands[0].Row() == 100 + segment);
            REQUIRE(row.channels[0][0].commands[0].Column() == 0);
            scheduler.Submit(std::move(row), cycle);
            REQUIRE(scheduler.CompletedVectors() == 0);
            REQUIRE_FALSE(scheduler.Running());
            for (auto* c : controllers) REQUIRE(c->PimRowIssued() == 0);
            for (; cycle < 3000 && scheduler.Stats().completed <
                                       uint64_t(segment + 1);
                 ++cycle) {
                scheduler.Tick(cycle);
                for (auto* c : controllers) c->ClockTick();
            }
            REQUIRE(scheduler.Stats().completed == uint64_t(segment + 1));
            REQUIRE(scheduler.CompletedVectors() == vectors);
            REQUIRE_FALSE(scheduler.Running());
            const auto start = scheduler.Stats().last_start;
            if (segment > 0) REQUIRE(start > previous_end);
            previous_end = start + (vectors - 1) * 2 + (local ? 0 : 1);
            REQUIRE(scheduler.Stats().last_end == previous_end);
            expected.push_back({start, origin, vectors});
        }
        REQUIRE(scheduler.Stats().launched == 3);
#ifdef CMD_TRACE
        CheckRowTrace(hw, local, expected);
#endif
        std::ifstream records(hw.output_dir + "hbnpu_rows.jsonl");
        REQUIRE(records.good());
        std::string line;
        size_t index = 0;
        while (std::getline(records, line)) {
            REQUIRE(index < expected.size());
            const auto record = nlohmann::json::parse(line);
            const auto& want = expected[index];
            REQUIRE(record.at("id").get<size_t>() == index);
            REQUIRE(record.at("vectors").get<int>() == want.vectors);
            REQUIRE(record.at("row").get<int>() == int(100 + index));
            REQUIRE(record.at("first_column").get<int>() == 0);
            REQUIRE(record.at("start_cycle").get<uint64_t>() == want.start);
            REQUIRE(record.at("last_issue_cycle").get<uint64_t>() ==
                    want.start + (want.vectors - 1) * 2 + (local ? 0 : 1));
            ++index;
        }
        REQUIRE(index == 3);
    }
}

TEST_CASE("HB-NPU row decomposition stops at physical row and tile tails",
          "[pim-row]") {
    Config hw("configs/HBM2_8Gb_x128_pim.ini", "/tmp");
    for (bool local : {false, true}) {
        auto c = Context(local);
        auto row = PimRowOperation::Build(hw, c, 17, 0, 65);
        REQUIRE(row.vectors == 15);
        REQUIRE(row.channels.size() == 8);
        REQUIRE(row.channels[0].front().commands.front().Column() == 17);
        REQUIRE(row.channels[0].back().commands.front().Column() == 31);
        REQUIRE(row.channels[0].back().commands.front().Row() == 100);
        REQUIRE(row.channels[0].size() == (local ? 15 : 30));
        auto tail = PimRowOperation::Build(hw, c, 64, 0, 65);
        REQUIRE(tail.vectors == 1);
        REQUIRE(tail.channels[0].front().commands.front().Row() == 102);
        REQUIRE(tail.channels[0].front().commands.front().Column() == 0);
    }
}
TEST_CASE("Rows reject refresh intervals with no usable post-refresh window",
          "[pim-row]") {
    char temp[] = "/tmp/hbnpu-row-window-XXXXXX";
    REQUIRE(mkdtemp(temp) != nullptr);
    Config hw("configs/HBM2_8Gb_x128_pim.ini", temp);
    hw.tREFI = 350;
    Timing timing(hw);
    std::vector<std::unique_ptr<Controller>> owned;
    std::vector<Controller*> controllers;
    for (int ch = 0; ch < hw.channels; ++ch) {
        owned.emplace_back(new Controller(ch, hw, timing));
        controllers.push_back(owned.back().get());
    }
    PimRowScheduler scheduler(hw, controllers);
    for (int cycle = 0; cycle < 340; ++cycle)
        for (auto* c : controllers) c->ClockTick();
    REQUIRE_THROWS_AS(
        scheduler.Submit(PimRowOperation::Build(hw, Context(false), 0, 0, 65),
                         340),
        std::invalid_argument);
    REQUIRE(scheduler.Stats().launched == 0);
}

TEST_CASE("A missed row beat is rejected rather than rescheduled",
          "[pim-row]") {
    char temp[] = "/tmp/hbnpu-row-fault-XXXXXX";
    REQUIRE(mkdtemp(temp) != nullptr);
    Config hw("configs/HBM2_8Gb_x128_pim.ini", temp);
    Timing timing(hw);
    std::vector<std::unique_ptr<Controller>> owned;
    std::vector<Controller*> controllers;
    for (int ch = 0; ch < hw.channels; ++ch) {
        owned.emplace_back(new Controller(ch, hw, timing));
        controllers.push_back(owned.back().get());
    }
    PimRowScheduler scheduler(hw, controllers);
    auto row = PimRowOperation::Build(hw, Context(false), 0, 0, 65);
    // Fault injection: close channel 3's first bank before its next scheduled
    // read. A permissive controller would insert ACT/tRCD and silently drift.
    auto command = row.channels[3][0].commands.front();
    command.cmd_type = CommandType::GH_READ_PRECHARGE;
    row.channels[3][0] = PimOperation(command, 0);
    scheduler.Submit(std::move(row), 0);
    REQUIRE_THROWS_WITH(
        ([&]() {
            for (int cycle = 0; cycle < 300; ++cycle) {
                scheduler.Tick(cycle);
                for (auto* c : controllers) c->ClockTick();
            }
        })(),
        "PIM row predictability violation: missed scheduled beat");
    REQUIRE(scheduler.Stats().completed == 0);
}
TEST_CASE(
    "Row consumption has exact cadence despite boundary and refresh delays",
    "[pim-row]") {
    for (bool local : {false, true}) {
        for (int scenario : {0, 1, 2, 3}) {
            char temp[] = "/tmp/hbnpu-row-test-XXXXXX";
            REQUIRE(mkdtemp(temp) != nullptr);
            Config hw("configs/HBM2_8Gb_x128_pim.ini", temp);
            hw.tREFI = 1000;
            if (scenario == 3) {
                hw.BL = 8;
                hw.burst_cycle = 4;
            }
            Timing timing(hw);
            std::vector<std::unique_ptr<Controller>> owned;
            std::vector<Controller*> controllers;
            for (int ch = 0; ch < hw.channels; ++ch) {
                owned.emplace_back(new Controller(ch, hw, timing));
                controllers.push_back(owned.back().get());
                controllers.back()->in_pim = true;
            }
            PimRowScheduler scheduler(hw, controllers);
            const int request = scenario == 1 ? 970 : 0;
            bool submitted = false;
            size_t issued = 0;
            for (int cycle = 0; cycle < 3000; ++cycle) {
                scheduler.Tick(cycle);
                if (cycle == request) {
                    if (scenario == 2) {
                        Command cmd(CommandType::GH_READ_PRECHARGE,
                                    Address(3, 0, 0, 0, 77, 0), 0);
                        controllers[3]->EnqueuePimOperation(
                            PimSource::WEIGHT, PimOperation(cmd, 200));
                    }
                    scheduler.Submit(
                        PimRowOperation::Build(hw, Context(local), 0, 0, 65),
                        cycle);
                    submitted = true;
                }
                for (auto* controller : controllers) controller->ClockTick();
                if (scheduler.Running()) {
                    size_t now = controllers[0]->PimRowIssued();
                    for (auto* controller : controllers)
                        REQUIRE(controller->PimRowIssued() == now);
                    if (now > issued) {
                        REQUIRE(now == issued + 1);
                        // Independently specified: GH A/B each cycle; LH every
                        // 2 cycles.
                        uint64_t offset =
                            scenario == 3
                                ? (local ? issued * 4
                                         : (issued / 2) * 4 + issued % 2)
                                : issued * (local ? 2 : 1);
                        REQUIRE(uint64_t(cycle) ==
                                scheduler.Stats().last_start + offset);
                        issued = now;
                    }
                }
                if (submitted && scheduler.Stats().completed == 1) break;
            }
            REQUIRE(scheduler.Stats().completed == 1);
            REQUIRE(scheduler.CompletedVectors() == (scenario == 3 ? 16 : 32));
            REQUIRE(issued ==
                    (scenario == 3 ? (local ? 16 : 32) : (local ? 32 : 64)));
            REQUIRE(scheduler.Stats().last_end - scheduler.Stats().last_start ==
                    (scenario == 3 ? (local ? 60 : 61) : (local ? 62 : 63)));
            if (scenario == 1) REQUIRE(scheduler.Stats().last_start >= 1260);
            if (scenario == 2) REQUIRE(scheduler.Stats().last_start >= 200);
        }
    }
}
