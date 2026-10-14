#include "cpu/bfs_cpu.hpp"

namespace cpu {

std::vector<uint32_t> bfsSerial(const graph::CSR& g, uint32_t source) {
    std::vector<uint32_t> dist(g.n, kUnreached);
    // A flat array as the FIFO: each vertex is enqueued at most once, so n
    // slots suffice and there is no deque bookkeeping.
    std::vector<uint32_t> queue(g.n);
    size_t head = 0, tail = 0;
    dist[source] = 0;
    queue[tail++] = source;
    while (head < tail) {
        uint32_t u = queue[head++];
        uint32_t du = dist[u] + 1;
        for (const uint32_t* p = g.begin(u); p != g.end(u); ++p) {
            if (dist[*p] == kUnreached) {
                dist[*p] = du;
                queue[tail++] = *p;
            }
        }
    }
    return dist;
}

} // namespace cpu
