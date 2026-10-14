#pragma once
// Compressed Sparse Row graph + the on-disk .bin format (PLAN.md 4.2).
//
// A CSR is a cheap-to-copy *view*: row_ptr/col_idx point into storage that is
// kept alive by a shared_ptr (either an mmap'd file or a pair of vectors).
// Copying a CSR shares the arrays; nothing is duplicated. Arrays are immutable.
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace graph {

enum : uint32_t {
    kFlagDirected = 1u << 0,
    kFlagSorted   = 1u << 1,
    kFlagDedup    = 1u << 2,
};

struct FileHeader {
    char     magic[8];      // "GGPUCSR1"
    uint32_t version;       // 1
    uint32_t flags;         // kFlag*
    uint64_t num_vertices;  // n
    uint64_t num_edges;     // m = entries in col_idx
};
static_assert(sizeof(FileHeader) == 32, "header layout must match the file format");

struct CSR {
    uint64_t n = 0;
    uint64_t m = 0;
    uint32_t flags = 0;
    const uint32_t* row_ptr = nullptr;  // n+1 entries
    const uint32_t* col_idx = nullptr;  // m entries
    std::shared_ptr<const void> storage;

    bool directed() const { return flags & kFlagDirected; }
    uint32_t degree(uint32_t v) const { return row_ptr[v + 1] - row_ptr[v]; }
    const uint32_t* begin(uint32_t v) const { return col_idx + row_ptr[v]; }
    const uint32_t* end(uint32_t v) const { return col_idx + row_ptr[v + 1]; }

    // Take ownership of in-memory arrays. row_ptr.size() must be n+1.
    static CSR fromVectors(std::vector<uint32_t> row_ptr, std::vector<uint32_t> col_idx,
                           uint32_t flags);
};

// Write a .bin file. Throws std::runtime_error on I/O failure.
void save(const CSR& g, const std::string& path);

// mmap a .bin file (read-only). Checks magic, version and sizes; throws on error.
CSR load(const std::string& path);

// Structural check: row_ptr monotone with row_ptr[0]=0, row_ptr[n]=m, all
// col_idx < n; if sorted/dedup flags are set, lists are strictly increasing
// and have no self-loops. Returns "" if valid, otherwise a description.
std::string validate(const CSR& g);

// Exact equality of n, m, flags and both arrays.
bool equal(const CSR& a, const CSR& b);

// Sort each adjacency list in place, drop duplicates and self-loops, compact
// col_idx and rebuild row_ptr. col_idx is NOT shrunk (callers use row_ptr[n]),
// so the converter avoids a second ~1 GB allocation on Orkut.
// Parallelized with OpenMP over vertices.
void compactAdjacency(std::vector<uint32_t>& row_ptr, std::vector<uint32_t>& col_idx);

// Build a CSR from an in-memory edge list (tests and generators only; the
// converter streams instead). Undirected: each edge is stored both ways.
CSR fromEdgeList(uint64_t n, const std::vector<std::pair<uint32_t, uint32_t>>& edges,
                 bool directed);

} // namespace graph
