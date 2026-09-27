// StructRender C++ runtime — one TensorRT engine, fixed shapes. KB §4 C++ Runtime, M10.
//
// Contract: nothing here throws into the caller. Every failure returns false and
// leaves a message in last_error(), so the game loop can hold the last valid
// frame or fall back to native RGB (§4: "renderer failure does not break gameplay").
#pragma once
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace structrender {

struct TensorInfo {
    std::string name;
    std::vector<long> shape;
    bool is_input = false;
    size_t elements() const;
};

class Engine {
public:
    Engine();
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    bool load(const std::string& engine_path);
    bool loaded() const;
    const std::vector<TensorInfo>& tensors() const;

    // Host float32 buffers, one per tensor in tensors() order (inputs read,
    // outputs written). Sizes must match elements(); checked, never assumed.
    bool infer(const std::vector<const float*>& inputs, const std::vector<float*>& outputs);

    // Device-side timing of enqueue+sync over n runs on the uploaded inputs, in ms.
    bool benchmark(int warmup, int iters, std::vector<float>& ms_out);

    const std::string& last_error() const;

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace structrender
