#pragma once
// Deterministic, fast PRNG (SplitMix64). Same seed -> same sequence on every
// run and platform, which is what fixed BFS sources and generators need.
#include <cstdint>

namespace util {

class Rng {
public:
    explicit Rng(uint64_t seed) : s_(seed) {}
    uint64_t next() {
        uint64_t z = (s_ += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    // Uniform in [0, bound) (Lemire's multiply-shift; bias is negligible here).
    uint64_t below(uint64_t bound) {
        return (uint64_t)(((unsigned __int128)next() * bound) >> 64);
    }
    // Uniform double in [0, 1).
    double uniform() { return (next() >> 11) * (1.0 / 9007199254740992.0); }

private:
    uint64_t s_;
};

} // namespace util
