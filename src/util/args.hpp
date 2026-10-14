#pragma once
// Minimal command-line parser: "--flag" and "--key value" (or "--key=value").
// Deliberately tiny; no external dependency.
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

namespace util {

class Args {
public:
    Args(int argc, char** argv) {
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if (a.rfind("--", 0) != 0) { positional_.push_back(a); continue; }
            a = a.substr(2);
            auto eq = a.find('=');
            if (eq != std::string::npos) {
                kv_[a.substr(0, eq)] = a.substr(eq + 1);
            } else if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
                kv_[a] = argv[++i];
            } else {
                kv_[a] = "";  // bare flag
            }
        }
    }

    bool has(const std::string& k) const { return kv_.count(k) != 0; }
    std::string get(const std::string& k, const std::string& def = "") const {
        auto it = kv_.find(k);
        return it == kv_.end() ? def : it->second;
    }
    long long getInt(const std::string& k, long long def) const {
        auto it = kv_.find(k);
        return (it == kv_.end() || it->second.empty()) ? def : std::atoll(it->second.c_str());
    }
    double getDouble(const std::string& k, double def) const {
        auto it = kv_.find(k);
        return (it == kv_.end() || it->second.empty()) ? def : std::atof(it->second.c_str());
    }
    const std::vector<std::string>& positional() const { return positional_; }

private:
    std::map<std::string, std::string> kv_;
    std::vector<std::string> positional_;
};

} // namespace util
