#pragma once
// Provenance fields for every results row: timestamp, git commit, chip, load.
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>
#include <sys/sysctl.h>

namespace util {

inline std::string timestamp() {
    std::time_t t = std::time(nullptr);
    char buf[32];
    std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%S", std::localtime(&t));
    return buf;
}

inline std::string chipName() {
    char buf[128] = {0};
    size_t len = sizeof buf;
    if (sysctlbyname("machdep.cpu.brand_string", buf, &len, nullptr, 0) != 0) return "unknown";
    return buf;
}

// Short commit hash, with "+" appended if tracked files have uncommitted
// changes (results from a dirty tree are not exactly reproducible).
inline std::string gitCommit() {
    auto run = [](const char* cmd) {
        std::string out;
        if (FILE* p = popen(cmd, "r")) {
            char buf[128];
            while (std::fgets(buf, sizeof buf, p)) out += buf;
            pclose(p);
        }
        while (!out.empty() && (out.back() == '\n' || out.back() == ' ')) out.pop_back();
        return out;
    };
    std::string h = run("git rev-parse --short HEAD 2>/dev/null");
    if (h.empty()) return "unknown";
    if (!run("git status --porcelain --untracked-files=no 2>/dev/null").empty()) h += "+";
    return h;
}

inline double loadAverage1() {
    double l[3];
    return getloadavg(l, 3) > 0 ? l[0] : -1.0;
}

} // namespace util
