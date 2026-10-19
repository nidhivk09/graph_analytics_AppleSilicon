// gexport: binary CSR (.bin) -> GAP Benchmark Suite serialized graph (.sg).
//   gexport <in.bin> <out.sg> [--sym]
//
// Phase 11 runs GAP's bfs and pr on exactly the graphs our variants see, so
// the comparison does not depend on two converters agreeing about duplicates
// and self-loops. GAP's .sg layout (gapbs src/writer.h, WriteSerializedGraph):
//   bool directed; int64 num_edges (adjacency entries); int64 num_nodes;
//   int64 offsets[n+1]; int32 neighbors[num_edges];
//   if directed: int64 in_offsets[n+1]; int32 in_neighbors[num_edges].
// --sym writes the symmetrized graph as undirected (what our BFS traverses).
// Without it, a directed graph is written with its transpose as the in-CSR
// (what our PageRank pulls over); an undirected graph is written as is.
#include "graph/csr.hpp"
#include "graph/transform.hpp"
#include "util/args.hpp"

#include <algorithm>
#include <cstdio>
#include <vector>

namespace {

// Offsets are 32-bit in our format and 64-bit in GAP's: widen in chunks so
// the export never holds a second full-size offsets array.
bool writeCsr(std::FILE* f, const graph::CSR& g) {
    std::vector<int64_t> buf;
    for (uint64_t i = 0; i <= g.n; i += 1 << 20) {
        uint64_t end = std::min<uint64_t>(g.n + 1, i + (1 << 20));
        buf.assign(g.row_ptr + i, g.row_ptr + end);
        if (std::fwrite(buf.data(), sizeof(int64_t), buf.size(), f) != buf.size()) return false;
    }
    // Vertex ids are < 2^31 for every dataset (checked in main), so uint32
    // ids are bit-identical to GAP's int32 NodeID.
    return std::fwrite(g.col_idx, sizeof(uint32_t), g.m, f) == g.m;
}

}  // namespace

int main(int argc, char** argv) {
    util::Args args(argc, argv);
    const auto& pos = args.positional();
    if (pos.size() != 2) {
        std::fprintf(stderr, "usage: gexport <in.bin> <out.sg> [--sym]\n");
        return 2;
    }
    try {
        graph::CSR g = graph::load(pos[0]);
        if (g.n >= (1ull << 31)) {
            std::fprintf(stderr, "gexport: %llu vertices do not fit GAP's int32 ids\n", (unsigned long long)g.n);
            return 1;
        }
        if (args.has("sym") && g.directed()) g = graph::symmetrize(g);
        const bool directed = !args.has("sym") && g.directed();

        std::FILE* f = std::fopen(pos[1].c_str(), "wb");
        if (!f) {
            std::perror(pos[1].c_str());
            return 1;
        }
        const int64_t m = (int64_t)g.m, n = (int64_t)g.n;
        bool ok = std::fwrite(&directed, sizeof(bool), 1, f) == 1 && std::fwrite(&m, sizeof m, 1, f) == 1 &&
                  std::fwrite(&n, sizeof n, 1, f) == 1 && writeCsr(f, g);
        if (ok && directed) ok = writeCsr(f, graph::transpose(g));
        ok = (std::fclose(f) == 0) && ok;
        if (!ok) {
            std::fprintf(stderr, "gexport: write to %s failed\n", pos[1].c_str());
            return 1;
        }
        std::printf("%s -> %s: %s, n %lld, %lld adjacency entries\n", pos[0].c_str(), pos[1].c_str(),
                    directed ? "directed (+ in-CSR)" : "undirected", (long long)n, (long long)m);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "gexport: %s\n", e.what());
        return 1;
    }
    return 0;
}
