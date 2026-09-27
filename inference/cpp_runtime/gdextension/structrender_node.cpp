#include "structrender_node.hpp"

#include <godot_cpp/core/class_db.hpp>

#include <chrono>
#include <cmath>

using namespace godot;

StructRenderer::StructRenderer() : engine_(new structrender::Engine) {}
StructRenderer::~StructRenderer() = default;

void StructRenderer::_bind_methods() {
    ClassDB::bind_method(D_METHOD("load_engine", "path"), &StructRenderer::load_engine);
    ClassDB::bind_method(D_METHOD("is_loaded"), &StructRenderer::is_loaded);
    ClassDB::bind_method(D_METHOD("render", "depth", "normal", "albedo", "width", "height"), &StructRenderer::render);
    ClassDB::bind_method(D_METHOD("last_error"), &StructRenderer::last_error);
    ClassDB::bind_method(D_METHOD("last_ms"), &StructRenderer::last_ms);
}

bool StructRenderer::load_engine(const String& path) {
    if (!engine_->load(path.utf8().get_data())) { err_ = String(engine_->last_error().c_str()); return false; }
    size_t nin = 0, nout = 0;
    for (const auto& t : engine_->tensors()) {
        if (t.is_input) { ++nin; in_.resize(t.elements()); }
        else { ++nout; out_.resize(t.elements()); }
    }
    if (nin != 1 || nout != 1) { err_ = "expected a 1-input/1-output engine"; return false; }
    err_ = "";
    return true;
}

bool StructRenderer::is_loaded() const { return engine_->loaded(); }
String StructRenderer::last_error() const { return err_; }
double StructRenderer::last_ms() const { return last_ms_; }

PackedByteArray StructRenderer::render(const PackedFloat32Array& depth, const PackedFloat32Array& normal,
                                       const PackedFloat32Array& albedo, int w, int h) {
    PackedByteArray rgb;
    auto t0 = std::chrono::steady_clock::now();
    if (!engine_->loaded()) { err_ = "no engine loaded"; return rgb; }
    const size_t hw = static_cast<size_t>(w) * h;
    if (in_.size() != 7 * hw || out_.size() != 3 * hw) { err_ = "resolution does not match the engine"; return rgb; }
    if (static_cast<size_t>(depth.size()) != hw || static_cast<size_t>(normal.size()) != 3 * hw ||
        static_cast<size_t>(albedo.size()) != 3 * hw) { err_ = "buffer sizes do not match width*height"; return rgb; }
    // renderer/training/dataset.py::build_input, channel for channel:
    //   [ 1/(1+depth), normal.xyz, albedo.rgb ]  as CHW float32.
    // Entity IDs and motion are deliberately not read (Decisions 8, 9).
    const float* d = depth.ptr(); const float* n = normal.ptr(); const float* a = albedo.ptr();
    for (size_t i = 0; i < hw; ++i) {
        in_[i] = 1.0f / (1.0f + d[i]);           // +inf background -> exactly 0
        for (int c = 0; c < 3; ++c) {
            in_[(1 + c) * hw + i] = n[i * 3 + c];
            in_[(4 + c) * hw + i] = a[i * 3 + c];
        }
    }
    if (!engine_->infer({in_.data()}, {out_.data()})) { err_ = String(engine_->last_error().c_str()); return rgb; }
    rgb.resize(static_cast<int64_t>(3 * hw));
    uint8_t* o = rgb.ptrw();
    for (size_t i = 0; i < hw; ++i)
        for (int c = 0; c < 3; ++c) {
            float v = out_[c * hw + i];
            if (!std::isfinite(v)) { err_ = "non-finite model output"; return PackedByteArray(); }
            v = v < 0.f ? 0.f : (v > 1.f ? 1.f : v);
            o[i * 3 + c] = static_cast<uint8_t>(std::lround(v * 255.f));
        }
    last_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    err_ = "";
    return rgb;
}
