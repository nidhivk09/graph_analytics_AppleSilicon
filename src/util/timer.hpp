#pragma once
// Wall-clock timer for host-side ("end-to-end") measurements.
#include <chrono>

namespace util {

class Timer {
public:
    Timer() { reset(); }
    void reset() { start_ = Clock::now(); }
    double ms() const {
        return std::chrono::duration<double, std::milli>(Clock::now() - start_).count();
    }

private:
    // steady_clock: monotonic, so NTP adjustments can't distort a measurement.
    using Clock = std::chrono::steady_clock;
    Clock::time_point start_;
};

} // namespace util
