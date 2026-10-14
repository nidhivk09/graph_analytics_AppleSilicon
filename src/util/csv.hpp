#pragma once
// Tiny CSV helpers. Results files are keyed by their first column (e.g. the
// dataset name) so re-running a tool replaces its rows instead of duplicating.
#include <fstream>
#include <string>
#include <vector>

namespace util {

// "data/web-Google.bin" -> "web-Google"
inline std::string stem(const std::string& path) {
    size_t s = path.find_last_of('/');
    std::string base = (s == std::string::npos) ? path : path.substr(s + 1);
    size_t d = base.find('.');
    return d == std::string::npos ? base : base.substr(0, d);
}

// Rewrite `path`: keep the header and every row whose first field != key, then
// append `rows`. Creates the file (with `header`) if missing.
inline void csvUpsert(const std::string& path, const std::string& header, const std::string& key,
                      const std::vector<std::string>& rows) {
    std::vector<std::string> keep;
    {
        std::ifstream in(path);
        std::string line;
        bool first = true;
        while (std::getline(in, line)) {
            if (first) { first = false; continue; }  // old header; replaced below
            if (line.empty()) continue;
            if (line.substr(0, line.find(',')) != key) keep.push_back(line);
        }
    }
    std::ofstream out(path, std::ios::trunc);
    out << header << '\n';
    for (auto& l : keep) out << l << '\n';
    for (auto& l : rows) out << l << '\n';
}

} // namespace util
