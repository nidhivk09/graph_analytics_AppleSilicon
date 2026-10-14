// gconvert: SNAP edge list (.txt.gz) -> binary CSR (.bin).
//   gconvert <in.txt.gz> <out.bin> --directed|--undirected [--csv results/convert_stats.csv]
// Appends/replaces a row in the CSV with timing and peak resident memory.
#include "graph/snap.hpp"
#include "util/args.hpp"
#include "util/csv.hpp"

#include <cstdio>
#include <sys/resource.h>

int main(int argc, char** argv) {
    util::Args args(argc, argv);
    const auto& pos = args.positional();
    if (pos.size() != 2 || args.has("directed") == args.has("undirected")) {
        std::fprintf(stderr, "usage: gconvert <in.txt.gz> <out.bin> --directed|--undirected [--csv path]\n");
        return 2;
    }
    const bool directed = args.has("directed");
    try {
        graph::ConvertStats st = graph::convertSnap(pos[0], pos[1], directed);
        struct rusage ru;
        getrusage(RUSAGE_SELF, &ru);
        double rssMb = ru.ru_maxrss / (1024.0 * 1024.0);  // macOS reports bytes

        std::printf("%s -> %s (%s)\n", pos[0].c_str(), pos[1].c_str(), directed ? "directed" : "undirected");
        std::printf("  lines %llu, raw max id %llu, n %llu\n", (unsigned long long)st.lines,
                    (unsigned long long)st.raw_max_id, (unsigned long long)st.n);
        std::printf("  entries %llu before dedup/self-loops, %llu after\n",
                    (unsigned long long)st.m_before, (unsigned long long)st.m);
        std::printf("  %.2f s, peak RSS %.1f MB\n", st.seconds, rssMb);

        std::string csv = args.get("csv", "results/convert_stats.csv");
        if (!csv.empty()) {
            std::string name = util::stem(pos[1]);
            char row[512];
            std::snprintf(row, sizeof row, "%s,%d,%llu,%llu,%llu,%llu,%.2f,%.1f", name.c_str(),
                          directed ? 1 : 0, (unsigned long long)st.lines, (unsigned long long)st.n,
                          (unsigned long long)st.m_before, (unsigned long long)st.m, st.seconds, rssMb);
            util::csvUpsert(csv, "dataset,directed,lines,n,m_before,m,seconds,peak_rss_mb", name, {row});
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "gconvert: %s\n", e.what());
        return 1;
    }
    return 0;
}
