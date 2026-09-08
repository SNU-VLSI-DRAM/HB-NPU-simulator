#ifndef __PIM_OPERATION_H
#define __PIM_OPERATION_H

#include <algorithm>
#include <set>
#include <stdexcept>
#include <vector>
#include "common.h"
#include "configuration.h"

namespace dramsim3 {
enum class PimOpcode {
    INVALID, LH_READ, LH_READ_PRECHARGE, LH_WRITE, LH_WRITE_PRECHARGE,
    GH_READ, GH_READ_PRECHARGE, GH_WRITE, GH_WRITE_PRECHARGE,
    GANG_ACT, GANG_PRE, PIM_ACT, PIM_PRE
};
enum class PimSource { WEIGHT, INPUT, OUTPUT };

// A queued data intent retains all targets while its prerequisites are issued.
// Its physical commands also retain the original tensor/hex addresses.
struct PimOperation {
    PimOpcode opcode = PimOpcode::INVALID;
    std::vector<Command> commands;
    uint64_t release_cycle = 0;
    PimOperation() = default;
    PimOperation(PimOpcode op, std::vector<Command> targets, uint64_t release = 0)
        : opcode(op), commands(std::move(targets)), release_cycle(release) {}
    explicit PimOperation(const Command& cmd, uint64_t release = 0)
        : commands{cmd}, release_cycle(release) {
        switch (cmd.cmd_type) {
            case CommandType::GH_READ: opcode = PimOpcode::GH_READ; break;
            case CommandType::GH_READ_PRECHARGE: opcode = PimOpcode::GH_READ_PRECHARGE; break;
            case CommandType::PIM_WRITE: opcode = PimOpcode::GH_WRITE; break;
            case CommandType::PIM_WRITE_PRECHARGE: opcode = PimOpcode::GH_WRITE_PRECHARGE; break;
            case CommandType::LH_READ: opcode = PimOpcode::LH_READ; break;
            case CommandType::LH_READ_PRECHARGE: opcode = PimOpcode::LH_READ_PRECHARGE; break;
            case CommandType::PIM_ACTIVATE: opcode = PimOpcode::PIM_ACT; break;
            case CommandType::PRECHARGE: opcode = PimOpcode::PIM_PRE; break;
            default: throw std::invalid_argument("Not a PIM command");
        }
    }
    bool IsValid() const { return opcode != PimOpcode::INVALID; }
    bool IsLocal() const { return opcode >= PimOpcode::LH_READ && opcode <= PimOpcode::LH_WRITE_PRECHARGE; }
    bool IsGlobal() const { return opcode >= PimOpcode::GH_READ && opcode <= PimOpcode::GH_WRITE_PRECHARGE; }
    bool IsData() const { return IsLocal() || IsGlobal(); }
    bool IsActivate() const { return opcode == PimOpcode::GANG_ACT || opcode == PimOpcode::PIM_ACT; }
    CommandType PhysicalType() const {
        switch (opcode) {
            case PimOpcode::LH_READ: return CommandType::LH_READ;
            case PimOpcode::LH_READ_PRECHARGE: return CommandType::LH_READ_PRECHARGE;
            case PimOpcode::LH_WRITE: case PimOpcode::GH_WRITE: return CommandType::PIM_WRITE;
            case PimOpcode::LH_WRITE_PRECHARGE: case PimOpcode::GH_WRITE_PRECHARGE: return CommandType::PIM_WRITE_PRECHARGE;
            case PimOpcode::GH_READ: return CommandType::GH_READ;
            case PimOpcode::GH_READ_PRECHARGE: return CommandType::GH_READ_PRECHARGE;
            case PimOpcode::GANG_ACT: case PimOpcode::PIM_ACT: return CommandType::PIM_ACTIVATE;
            case PimOpcode::GANG_PRE: case PimOpcode::PIM_PRE: return CommandType::PRECHARGE;
            default: return CommandType::SIZE;
        }
    }
    const char* Name() const {
        static const char* names[] = {"INVALID", "LH_READ", "LH_READ_PRECHARGE", "LH_WRITE", "LH_WRITE_PRECHARGE", "GH_READ", "GH_READ_PRECHARGE", "GH_WRITE", "GH_WRITE_PRECHARGE", "GANG_ACT", "GANG_PRE", "PIM_ACT", "PIM_PRE"};
        return names[static_cast<int>(opcode)];
    }
    void Validate(const Config& config) const {
        const size_t width = IsLocal() ? config.banks :
            (opcode == PimOpcode::GANG_ACT || opcode == PimOpcode::GANG_PRE ? 4 : 1);
        if (!IsValid() || commands.size() != width)
            throw std::invalid_argument("Invalid PIM operation width");
        const auto& first = commands.front();
        std::set<std::pair<int,int>> banks;
        for (const auto& cmd : commands) {
            if (cmd.cmd_type != PhysicalType() || cmd.Channel() != first.Channel() ||
                cmd.Rank() != first.Rank() || cmd.Row() != first.Row() ||
                cmd.Column() != first.Column() || cmd.Channel() < 0 ||
                cmd.Channel() >= config.channels || cmd.Rank() < 0 ||
                cmd.Rank() >= config.ranks || cmd.Bankgroup() < 0 ||
                cmd.Bankgroup() >= config.bankgroups || cmd.Bank() < 0 ||
                cmd.Bank() >= config.banks_per_group ||
                !banks.emplace(cmd.Bankgroup(),cmd.Bank()).second)
                throw std::invalid_argument("Invalid PIM operation target");
        }
    }
};
}  // namespace dramsim3
#endif
