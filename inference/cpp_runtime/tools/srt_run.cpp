// srt_run <engine> <in0.bin> [in1.bin ...] --out <out.bin> [--bench N]
// Runs one inference on raw float32 inputs (tensor order), writes the first
// output, optionally benchmarks. Exit 0 on success, 1 on any failure.
#include "structrender/engine.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static bool read_all(const std::string& p, std::vector<float>& v, size_t n) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return false;
    v.resize(n);
    f.read(reinterpret_cast<char*>(v.data()), n * sizeof(float));
    return static_cast<size_t>(f.gcount()) == n * sizeof(float);
}

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: srt_run <engine> <in.bin>... --out <out.bin> [--bench N]\n"); return 1; }
    std::vector<std::string> ins; std::string out; int bench = 0;
    for (int i = 2; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
        else if (!std::strcmp(argv[i], "--bench") && i + 1 < argc) bench = std::atoi(argv[++i]);
        else ins.push_back(argv[i]);
    }
    structrender::Engine e;
    if (!e.load(argv[1])) { std::fprintf(stderr, "load: %s\n", e.last_error().c_str()); return 1; }
    std::vector<std::vector<float>> in_bufs, out_bufs; std::vector<const float*> ip; std::vector<float*> op;
    size_t k = 0;
    for (const auto& t : e.tensors()) {
        std::printf("tensor %-10s %-6s [", t.name.c_str(), t.is_input ? "in" : "out");
        for (size_t d = 0; d < t.shape.size(); ++d) std::printf("%s%ld", d ? "," : "", t.shape[d]);
        std::printf("]\n");
        if (t.is_input) {
            if (k >= ins.size()) { std::fprintf(stderr, "missing input file for %s\n", t.name.c_str()); return 1; }
            in_bufs.emplace_back();
            if (!read_all(ins[k++], in_bufs.back(), t.elements())) { std::fprintf(stderr, "bad input file for %s\n", t.name.c_str()); return 1; }
        } else {
            out_bufs.emplace_back(t.elements());
        }
    }
    for (auto& b : in_bufs) ip.push_back(b.data());
    for (auto& b : out_bufs) op.push_back(b.data());
    if (!e.infer(ip, op)) { std::fprintf(stderr, "infer: %s\n", e.last_error().c_str()); return 1; }
    if (!out.empty()) {
        std::ofstream f(out, std::ios::binary);
        f.write(reinterpret_cast<const char*>(out_bufs[0].data()), out_bufs[0].size() * sizeof(float));
    }
    if (bench > 0) {
        std::vector<float> ms;
        if (!e.benchmark(20, bench, ms)) { std::fprintf(stderr, "bench: %s\n", e.last_error().c_str()); return 1; }
        std::sort(ms.begin(), ms.end());
        std::printf("bench n=%d median %.3f ms  p95 %.3f ms  min %.3f ms\n", bench,
                    ms[ms.size() / 2], ms[static_cast<size_t>(ms.size() * 0.95)], ms.front());
    }
    std::printf("ok\n");
    return 0;
}
