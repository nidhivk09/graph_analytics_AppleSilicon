#include "graph/csr.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace graph {

static const char kMagic[8] = {'G', 'G', 'P', 'U', 'C', 'S', 'R', '1'};

CSR CSR::fromVectors(std::vector<uint32_t> rp, std::vector<uint32_t> ci, uint32_t flags) {
    if (rp.empty()) throw std::runtime_error("fromVectors: row_ptr must have n+1 entries");
    auto store = std::make_shared<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>>(
        std::move(rp), std::move(ci));
    CSR g;
    g.n = store->first.size() - 1;
    g.m = store->first.back();  // col_idx may be longer than m (see compactAdjacency)
    g.flags = flags;
    g.row_ptr = store->first.data();
    g.col_idx = store->second.data();
    g.storage = store;
    return g;
}

void save(const CSR& g, const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write " + path);
    FileHeader h{};
    std::memcpy(h.magic, kMagic, 8);
    h.version = 1;
    h.flags = g.flags;
    h.num_vertices = g.n;
    h.num_edges = g.m;
    bool ok = std::fwrite(&h, sizeof h, 1, f) == 1 &&
              std::fwrite(g.row_ptr, 4, g.n + 1, f) == g.n + 1 &&
              (g.m == 0 || std::fwrite(g.col_idx, 4, g.m, f) == g.m);
    ok = (std::fclose(f) == 0) && ok;
    if (!ok) throw std::runtime_error("short write to " + path);
}

namespace {
struct Mapping {
    void* base = nullptr;
    size_t len = 0;
    ~Mapping() { if (base) munmap(base, len); }
};
} // namespace

CSR load(const std::string& path) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("cannot open " + path);
    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); throw std::runtime_error("cannot stat " + path); }
    size_t len = (size_t)st.st_size;
    if (len < sizeof(FileHeader)) { close(fd); throw std::runtime_error(path + ": too small"); }

    auto map = std::make_shared<Mapping>();
    map->base = mmap(nullptr, len, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);  // the mapping stays valid after close
    if (map->base == MAP_FAILED) { map->base = nullptr; throw std::runtime_error("mmap failed: " + path); }
    map->len = len;

    const auto* h = static_cast<const FileHeader*>(map->base);
    if (std::memcmp(h->magic, kMagic, 8) != 0) throw std::runtime_error(path + ": bad magic");
    if (h->version != 1) throw std::runtime_error(path + ": unsupported version");
    uint64_t expect = sizeof(FileHeader) + 4 * (h->num_vertices + 1) + 4 * h->num_edges;
    if (expect != len) throw std::runtime_error(path + ": size does not match header");

    CSR g;
    g.n = h->num_vertices;
    g.m = h->num_edges;
    g.flags = h->flags;
    g.row_ptr = reinterpret_cast<const uint32_t*>(static_cast<const char*>(map->base) + sizeof(FileHeader));
    g.col_idx = g.row_ptr + g.n + 1;
    // Benchmarks touch every byte anyway; ask the kernel to read ahead.
    madvise(map->base, len, MADV_WILLNEED);
    g.storage = map;
    return g;
}

std::string validate(const CSR& g) {
    if (g.m >= (1ull << 32)) return "m does not fit in uint32";
    if (g.row_ptr[0] != 0) return "row_ptr[0] != 0";
    if (g.row_ptr[g.n] != g.m) return "row_ptr[n] != m";
    const bool strict = (g.flags & kFlagSorted) && (g.flags & kFlagDedup);
    for (uint64_t v = 0; v < g.n; ++v) {
        uint32_t b = g.row_ptr[v], e = g.row_ptr[v + 1];
        if (e < b) return "row_ptr decreases at v=" + std::to_string(v);
        for (uint32_t i = b; i < e; ++i) {
            uint32_t u = g.col_idx[i];
            if (u >= g.n) return "col_idx out of range at v=" + std::to_string(v);
            if (strict) {
                if (u == v) return "self-loop at v=" + std::to_string(v);
                if (i > b && g.col_idx[i - 1] >= u) return "list not strictly sorted at v=" + std::to_string(v);
            }
        }
    }
    return "";
}

bool equal(const CSR& a, const CSR& b) {
    return a.n == b.n && a.m == b.m && a.flags == b.flags &&
           std::equal(a.row_ptr, a.row_ptr + a.n + 1, b.row_ptr) &&
           std::equal(a.col_idx, a.col_idx + a.m, b.col_idx);
}

void compactAdjacency(std::vector<uint32_t>& rp, std::vector<uint32_t>& ci) {
    const int64_t n = (int64_t)rp.size() - 1;
    std::vector<uint32_t> newDeg(n);

    // Phase A (parallel): each list is sorted and deduplicated inside its own
    // segment, results packed at the segment front. Dynamic scheduling because
    // power-law graphs put most of the work in a few hub lists.
#pragma omp parallel for schedule(dynamic, 1024)
    for (int64_t v = 0; v < n; ++v) {
        uint32_t* b = ci.data() + rp[v];
        uint32_t* e = ci.data() + rp[v + 1];
        std::sort(b, e);
        uint32_t* out = b;
        for (uint32_t* p = b; p != e; ++p) {
            if (*p == (uint32_t)v) continue;                 // self-loop
            if (out != b && out[-1] == *p) continue;         // duplicate
            *out++ = *p;
        }
        newDeg[v] = (uint32_t)(out - b);
    }

    // Phase B (serial): slide segments left. New offsets never exceed old ones,
    // so an in-order memmove never overwrites unread data.
    uint64_t pos = 0;
    for (int64_t v = 0; v < n; ++v) {
        uint32_t start = rp[v];
        rp[v] = (uint32_t)pos;
        if (newDeg[v] && start != pos)
            std::memmove(ci.data() + pos, ci.data() + start, 4ull * newDeg[v]);
        pos += newDeg[v];
    }
    rp[n] = (uint32_t)pos;
}

CSR fromEdgeList(uint64_t n, const std::vector<std::pair<uint32_t, uint32_t>>& edges, bool directed) {
    std::vector<uint32_t> rp(n + 1, 0);
    for (auto [u, v] : edges) {
        rp[u + 1]++;
        if (!directed) rp[v + 1]++;
    }
    for (uint64_t i = 0; i < n; ++i) rp[i + 1] += rp[i];
    std::vector<uint32_t> ci(rp[n]);
    std::vector<uint32_t> cur(rp.begin(), rp.end() - 1);
    for (auto [u, v] : edges) {
        ci[cur[u]++] = v;
        if (!directed) ci[cur[v]++] = u;
    }
    compactAdjacency(rp, ci);
    ci.resize(rp[n]);
    return CSR::fromVectors(std::move(rp), std::move(ci),
                            (directed ? kFlagDirected : 0) | kFlagSorted | kFlagDedup);
}

} // namespace graph
