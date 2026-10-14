#pragma once
// CPU BFS variants (PLAN.md 5.1). cpu_serial is the reference every other
// variant is checked against. All variants write dist[v] (kUnreached if v is
// not reachable) into a caller-provided array of g.n entries, so the output
// allocation is outside the timed region while initialization is inside.
#include "graph/csr.hpp"
#include "util/trace.hpp"

#include <string>
#include <vector>

namespace cpu {

constexpr uint32_t kUnreached = 0xFFFFFFFFu;

enum class Schedule { Static, Dynamic64, Guided };
// Accepts "static", "dynamic,64" (or "dynamic"), "guided"; throws otherwise.
Schedule parseSchedule(const std::string& s);
const char* scheduleName(Schedule s);  // CSV-safe: "static", "dynamic/64", "guided"

struct BfsOptions {
    int threads = 0;                      // 0 = OpenMP default (all cores)
    Schedule schedule = Schedule::Dynamic64;  // cpu_omp only
    double alpha = 15.0;                  // cpu_do: switch TD->BU if m_f > m_u / alpha
    double beta = 18.0;                   // cpu_do: switch BU->TD if n_f < n / beta (and shrinking)
};

// Queue-based serial BFS.
void bfsSerial(const graph::CSR& g, uint32_t source, uint32_t* dist);
std::vector<uint32_t> bfsSerial(const graph::CSR& g, uint32_t source);  // convenience

// Level-synchronous top-down OpenMP BFS with configurable schedule.
void bfsOmp(const graph::CSR& g, uint32_t source, uint32_t* dist, const BfsOptions& opt,
            std::vector<util::BfsLevel>* trace = nullptr);

// Direction-optimizing BFS (Beamer et al., SC'12; heuristic as in GAP).
// g must be symmetric (bottom-up scans a vertex's neighbors as its parents).
void bfsDO(const graph::CSR& g, uint32_t source, uint32_t* dist, const BfsOptions& opt,
           std::vector<util::BfsLevel>* trace = nullptr);

} // namespace cpu
