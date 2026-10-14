#include "graph/snap.hpp"
#include "graph/csr.hpp"
#include "util/timer.hpp"

#include <cstring>
#include <stdexcept>
#include <vector>
#include <zlib.h>

namespace graph {

namespace {

// Calls fn(u, v) for every "u v" line of a (possibly gzipped) edge list.
// Lines starting with '#' or '%' are comments. Reads 4 MB chunks with gzread
// and parses by hand: gzgets + sscanf is several times slower on 100M lines.
template <class Fn>
uint64_t forEachEdge(const std::string& path, Fn&& fn) {
    gzFile f = gzopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + path);
    gzbuffer(f, 1 << 20);

    const size_t kChunk = 4 << 20;
    std::vector<char> buf(kChunk + 1);
    size_t have = 0;  // bytes carried over from the previous chunk (partial line)
    uint64_t lines = 0, lineNo = 0;
    bool eof = false;

    while (!eof) {
        int got = gzread(f, buf.data() + have, (unsigned)(kChunk - have));
        if (got < 0) { gzclose(f); throw std::runtime_error("gzread failed on " + path); }
        if (got == 0) eof = true;
        size_t len = have + (size_t)got;
        if (eof && len > 0 && buf[len - 1] != '\n') buf[len++] = '\n';  // last line w/o newline

        // Only parse up to the last complete line.
        size_t end = len;
        while (end > 0 && buf[end - 1] != '\n') --end;
        if (end == 0 && !eof && len == kChunk) {
            gzclose(f);
            throw std::runtime_error("line longer than 4 MB in " + path);
        }

        const char* p = buf.data();
        const char* stop = buf.data() + end;
        while (p < stop) {
            ++lineNo;
            while (p < stop && (*p == ' ' || *p == '\t' || *p == '\r')) ++p;
            if (*p == '\n') { ++p; continue; }                     // blank line
            if (*p == '#' || *p == '%') {                          // comment
                while (*p != '\n') ++p;
                ++p;
                continue;
            }
            uint64_t ids[2];
            for (int k = 0; k < 2; ++k) {
                while (*p == ' ' || *p == '\t') ++p;
                if (*p < '0' || *p > '9') {
                    gzclose(f);
                    throw std::runtime_error(path + ": malformed line " + std::to_string(lineNo));
                }
                uint64_t x = 0;
                while (*p >= '0' && *p <= '9') x = x * 10 + (uint64_t)(*p++ - '0');
                // Raw IDs index per-ID arrays; 0xFFFFFFFF is reserved as "absent".
                if (x >= 0xFFFFFFFFull) {
                    gzclose(f);
                    throw std::runtime_error(path + ": vertex ID too large on line " + std::to_string(lineNo));
                }
                ids[k] = x;
            }
            while (*p != '\n') ++p;  // ignore trailing columns (e.g. weights)
            ++p;
            fn((uint32_t)ids[0], (uint32_t)ids[1]);
            ++lines;
        }
        have = len - end;
        std::memmove(buf.data(), buf.data() + end, have);
    }
    gzclose(f);
    return lines;
}

} // namespace

ConvertStats convertSnap(const std::string& in, const std::string& out, bool directed) {
    util::Timer timer;
    ConvertStats st;

    // ---- Pass 1: which raw IDs exist, and how many entries each will own.
    // Arrays are indexed by *raw* ID (max ~5M here, so a few tens of MB at most)
    // and grown on demand because the max ID is unknown until the end.
    std::vector<uint32_t> rawDeg;
    std::vector<uint8_t> seen;
    auto touch = [&](uint32_t id) {
        if (id >= rawDeg.size()) {
            size_t sz = std::max<size_t>((size_t)id + 1, rawDeg.size() + rawDeg.size() / 2);
            rawDeg.resize(sz, 0);
            seen.resize(sz, 0);
        }
        seen[id] = 1;
    };
    uint64_t total = 0;
    st.lines = forEachEdge(in, [&](uint32_t u, uint32_t v) {
        touch(u);
        touch(v);
        rawDeg[u]++;
        if (!directed) rawDeg[v]++;
        total += directed ? 1 : 2;
    });
    if (total >= 0xFFFFFFFFull) throw std::runtime_error("too many edges for uint32 offsets");
    st.m_before = total;

    // ---- Dense relabeling in increasing raw-ID order. Preserving the input's
    // relative order matters: it *is* the "original" ordering of Phase 9.
    std::vector<uint32_t> remap(rawDeg.size(), 0xFFFFFFFFu);
    uint32_t n = 0;
    for (size_t id = 0; id < rawDeg.size(); ++id)
        if (seen[id]) { remap[id] = n++; st.raw_max_id = id; }
    seen = {};
    st.n = n;

    // ---- Prefix sum degrees -> row_ptr (exclusive scan).
    std::vector<uint32_t> rp(n + 1, 0);
    for (size_t id = 0; id < rawDeg.size(); ++id)
        if (remap[id] != 0xFFFFFFFFu) rp[remap[id] + 1] = rawDeg[id];
    rawDeg = {};
    for (uint32_t v = 0; v < n; ++v) rp[v + 1] += rp[v];

    // ---- Pass 2: scatter neighbors through per-vertex cursors.
    std::vector<uint32_t> ci(total);
    std::vector<uint32_t> cur(rp.begin(), rp.end() - 1);
    forEachEdge(in, [&](uint32_t u, uint32_t v) {
        uint32_t a = remap[u], b = remap[v];
        ci[cur[a]++] = b;
        if (!directed) ci[cur[b]++] = a;
    });
    cur = {};
    remap = {};

    // ---- Sort, dedup, drop self-loops, compact (OpenMP), then write.
    compactAdjacency(rp, ci);
    st.m = rp[n];
    CSR g = CSR::fromVectors(std::move(rp), std::move(ci),
                             (directed ? kFlagDirected : 0) | kFlagSorted | kFlagDedup);
    save(g, out);
    st.seconds = timer.ms() / 1e3;
    return st;
}

} // namespace graph
