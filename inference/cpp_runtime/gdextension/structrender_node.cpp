#include "structrender_node.hpp"

#include <godot_cpp/core/class_db.hpp>

#include <chrono>
#include <cmath>
#include <cstring>

using namespace godot;

StructRenderer::StructRenderer() : engine_(new structrender::Engine) {}
StructRenderer::~StructRenderer() = default;

void StructRenderer::_bind_methods() {
    ClassDB::bind_method(D_METHOD("load_engine", "path"), &StructRenderer::load_engine);
    ClassDB::bind_method(D_METHOD("is_loaded"), &StructRenderer::is_loaded);
    ClassDB::bind_method(D_METHOD("render", "depth", "normal", "albedo", "width", "height"), &StructRenderer::render);
    ClassDB::bind_method(D_METHOD("render_images", "depth", "normal", "albedo"), &StructRenderer::render_images);
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

static inline float half_to_float(uint16_t hv) {
    const uint32_t sign = (hv & 0x8000u) << 16, exp = (hv >> 10) & 0x1Fu, man = hv & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) bits = sign;
        else { int e = -1; uint32_t m = man; do { ++e; m <<= 1; } while (!(m & 0x400u));
               bits = sign | ((127 - 15 - e) << 23) | ((m & 0x3FFu) << 13); }
    } else if (exp == 31) bits = sign | 0x7F800000u | (man << 13);
    else bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
    float f; std::memcpy(&f, &bits, 4); return f;
}

// Channel layout of a pass image. The exporter's SubViewport returns FORMAT_RGBH
// (3 half-float channels, measured — not RGBA as first assumed); the float and
// 4-channel variants are accepted too. Anything else is refused, never guessed.
static bool pixel_reader(const Ref<Image>& im, int& stride, bool& is_half) {
    if (im.is_null()) return false;
    switch (im->get_format()) {
        case Image::FORMAT_RGBH:  stride = 3; is_half = true; return true;
        case Image::FORMAT_RGBF:  stride = 3; is_half = false; return true;
        case Image::FORMAT_RGBAH: stride = 4; is_half = true; return true;
        case Image::FORMAT_RGBAF: stride = 4; is_half = false; return true;
        default: return false;
    }
}

PackedByteArray StructRenderer::render_images(const Ref<Image>& depth, const Ref<Image>& normal, const Ref<Image>& albedo) {
    auto t0 = std::chrono::steady_clock::now();
    int sd, sn, sa; bool hd, hn, ha;
    if (!pixel_reader(depth, sd, hd) || !pixel_reader(normal, sn, hn) || !pixel_reader(albedo, sa, ha)) {
        err_ = "pass images must be RGBH, RGBF, RGBAH or RGBAF"; return PackedByteArray(); }
    const int w = depth->get_width(), h = depth->get_height();
    if (normal->get_width() != w || albedo->get_width() != w || normal->get_height() != h || albedo->get_height() != h) {
        err_ = "pass images differ in size"; return PackedByteArray(); }
    const size_t hw = static_cast<size_t>(w) * h;
    if (!engine_->loaded() || in_.size() != 7 * hw) { err_ = "engine not loaded or resolution mismatch"; return PackedByteArray(); }
    PackedByteArray bd = depth->get_data(), bn = normal->get_data(), ba = albedo->get_data();
    auto rd = [](const PackedByteArray& b, bool half, size_t idx) -> float {
        if (half) { uint16_t v; std::memcpy(&v, b.ptr() + idx * 2, 2); return half_to_float(v); }
        float v; std::memcpy(&v, b.ptr() + idx * 4, 4); return v;
    };
    for (size_t i = 0; i < hw; ++i) {
        const float dr = rd(bd, hd, i * sd);
        if (dr <= 0.0f) {                    // background: exactly export_frame's rule
            in_[i] = 0.0f;                   // 1/(1+inf)
            for (int c = 0; c < 3; ++c) { in_[(1 + c) * hw + i] = 0.0f; in_[(4 + c) * hw + i] = 0.0f; }
            continue;
        }
        in_[i] = 1.0f / (1.0f + dr);
        for (int c = 0; c < 3; ++c) {
            in_[(1 + c) * hw + i] = rd(bn, hn, i * sn + c);
            in_[(4 + c) * hw + i] = rd(ba, ha, i * sa + c);
        }
    }
    PackedByteArray out = run_from_input(w, h);
    last_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return out;
}

PackedByteArray StructRenderer::run_from_input(int w, int h) {
    const size_t hw = static_cast<size_t>(w) * h;
    PackedByteArray rgb;
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
    err_ = "";
    return rgb;
}

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
    PackedByteArray out = run_from_input(w, h);
    last_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return out;
}
