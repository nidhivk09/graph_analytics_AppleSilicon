#pragma once
// Streaming SNAP edge-list (.txt or .txt.gz) -> .bin converter (PLAN.md 4.3).
// Kept as a library (not just inside gconvert) so unit tests can drive it.
#include <cstdint>
#include <string>

namespace graph {

struct ConvertStats {
    uint64_t lines = 0;        // edge lines parsed (comments excluded)
    uint64_t raw_max_id = 0;   // largest vertex ID in the input
    uint64_t n = 0;            // vertices after relabeling
    uint64_t m_before = 0;     // col_idx entries before dedup/self-loop removal
    uint64_t m = 0;            // final col_idx entries
    double seconds = 0;
};

// Two streaming passes over `in` (zlib reads plain text too), never holding an
// edge list in memory. Peak memory ~ 4*m_before + O(max raw ID).
// directed=false: every input line u v yields both u->v and v->u.
// Throws std::runtime_error on malformed input or I/O failure.
ConvertStats convertSnap(const std::string& in, const std::string& out, bool directed);

} // namespace graph
