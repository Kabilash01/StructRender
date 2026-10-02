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
    ClassDB::bind_method(D_METHOD("load_temporal_engines", "paths"), &StructRenderer::load_temporal_engines);
    ClassDB::bind_method(D_METHOD("set_style", "style"), &StructRenderer::set_style);
    ClassDB::bind_method(D_METHOD("get_style"), &StructRenderer::get_style);
    ClassDB::bind_method(D_METHOD("reset_history"), &StructRenderer::reset_history);
    ClassDB::bind_method(D_METHOD("render_temporal", "depth", "normal", "albedo", "ids", "cam", "proj", "entity_xforms"),
                         &StructRenderer::render_temporal);
    ClassDB::bind_method(D_METHOD("last_motion"), &StructRenderer::last_motion);
    ClassDB::bind_method(D_METHOD("last_no_prev_pixels"), &StructRenderer::last_no_prev_pixels);
    ClassDB::bind_method(D_METHOD("last_motion_valid"), &StructRenderer::last_motion_valid);
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

// --------------------------------------------------------------------------- //
// TEMPORAL live path — two-style B with history (KB Decision 97)
// --------------------------------------------------------------------------- //
static const char* kTIn[] = {"x_t", "x_prev", "feat_prev", "motion", "no_prev", "motion_valid"};
static const char* kTOut[] = {"rgb", "feat"};

bool StructRenderer::load_temporal_engines(const PackedStringArray& paths) {
    tengines_.clear();
    for (int64_t k = 0; k < paths.size(); ++k) {
        auto e = std::make_unique<structrender::Engine>();
        if (!e->load(paths[k].utf8().get_data())) { err_ = String(e->last_error().c_str()); tengines_.clear(); return false; }
        // Every engine must have exactly the LiveStep contract, by name.
        size_t found = 0;
        for (const auto& t : e->tensors()) {
            for (const char* n : kTIn) if (t.is_input && t.name == n) ++found;
            for (const char* n : kTOut) if (!t.is_input && t.name == n) ++found;
            if (t.name == "x_t") { x_.assign(t.elements(), 0.f); x_prev_.assign(t.elements(), 0.f); }
            if (t.name == "feat_prev") { feat_prev_.assign(t.elements(), 0.f); feat_.assign(t.elements(), 0.f); }
            if (t.name == "motion") { motion_.assign(t.elements(), 0.f); motion_hwc_.assign(t.elements(), 0.f); }
            if (t.name == "no_prev") no_prev_.assign(t.elements(), 0.f);
            if (t.name == "rgb") rgbf_.assign(t.elements(), 0.f);
        }
        if (found != 8 || e->tensors().size() != 8) { err_ = "not a LiveStep engine (expected 6 named inputs, 2 outputs)"; tengines_.clear(); return false; }
        tengines_.push_back(std::move(e));
    }
    mvalid_.assign(1, 0.f);
    style_ = 0;
    reset_history();
    err_ = "";
    return !tengines_.empty();
}

void StructRenderer::set_style(int style) { if (style >= 0 && style < static_cast<int>(tengines_.size())) style_ = style; }
int StructRenderer::get_style() const { return style_; }
void StructRenderer::reset_history() { prev_valid_ = false; prev_xforms_.clear(); }
PackedFloat32Array StructRenderer::last_motion() const {
    PackedFloat32Array a; a.resize(static_cast<int64_t>(motion_hwc_.size()));
    if (!motion_hwc_.empty()) std::memcpy(a.ptrw(), motion_hwc_.data(), motion_hwc_.size() * sizeof(float));
    return a;
}
int StructRenderer::last_no_prev_pixels() const { return no_prev_count_; }
bool StructRenderer::last_motion_valid() const { return last_mvalid_; }

bool StructRenderer::decode_structure(const Ref<Image>& depth, const Ref<Image>& normal, const Ref<Image>& albedo,
                                      std::vector<float>& x, std::vector<float>& depth_m) {
    // Identical decode to render_images (export_frame's per-pixel rule).
    int sd, sn, sa; bool hd, hn, ha;
    if (!pixel_reader(depth, sd, hd) || !pixel_reader(normal, sn, hn) || !pixel_reader(albedo, sa, ha)) {
        err_ = "pass images must be RGBH, RGBF, RGBAH or RGBAF"; return false; }
    const int w = depth->get_width(), h = depth->get_height();
    const size_t hw = static_cast<size_t>(w) * h;
    if (x.size() != 7 * hw) { err_ = "resolution does not match the engine"; return false; }
    PackedByteArray bd = depth->get_data(), bn = normal->get_data(), ba = albedo->get_data();
    auto rd = [](const PackedByteArray& b, bool half, size_t idx) -> float {
        if (half) { uint16_t v; std::memcpy(&v, b.ptr() + idx * 2, 2); return half_to_float(v); }
        float v; std::memcpy(&v, b.ptr() + idx * 4, 4); return v;
    };
    depth_m.assign(hw, INFINITY);
    for (size_t i = 0; i < hw; ++i) {
        const float dr = rd(bd, hd, i * sd);
        if (dr <= 0.0f) {
            x[i] = 0.0f;
            for (int c = 0; c < 3; ++c) { x[(1 + c) * hw + i] = 0.0f; x[(4 + c) * hw + i] = 0.0f; }
            continue;
        }
        depth_m[i] = dr;
        x[i] = 1.0f / (1.0f + dr);
        for (int c = 0; c < 3; ++c) {
            x[(1 + c) * hw + i] = rd(bn, hn, i * sn + c);
            x[(4 + c) * hw + i] = rd(ba, ha, i * sa + c);
        }
    }
    return true;
}

PackedByteArray StructRenderer::render_temporal(const Ref<Image>& depth, const Ref<Image>& normal, const Ref<Image>& albedo,
                                                const Ref<Image>& ids, const Transform3D& cam, const Projection& proj,
                                                const Dictionary& entity_xforms) {
    auto t0 = std::chrono::steady_clock::now();
    if (tengines_.empty()) { err_ = "no temporal engines loaded"; return PackedByteArray(); }
    if (!decode_structure(depth, normal, albedo, x_, depth_m_)) { prev_valid_ = false; return PackedByteArray(); }
    const int W = depth->get_width(), H = depth->get_height();
    const size_t hw = static_cast<size_t>(W) * H;

    // entity ids, decoded exactly as export_frame: r + g*256 + b*65536 (0 = none)
    int si; bool hi;
    if (!pixel_reader(ids, si, hi) || ids->get_width() != W || ids->get_height() != H) {
        err_ = "entity-id pass missing or wrong size"; prev_valid_ = false; return PackedByteArray(); }
    {
        PackedByteArray bi = ids->get_data();
        auto rd = [&](size_t idx) -> float {
            if (hi) { uint16_t v; std::memcpy(&v, bi.ptr() + idx * 2, 2); return half_to_float(v); }
            float v; std::memcpy(&v, bi.ptr() + idx * 4, 4); return v; };
        ids_.assign(hw, 0);
        for (size_t i = 0; i < hw; ++i) {
            if (std::isinf(depth_m_[i])) continue;      // background: id 0, as export_frame
            ids_[i] = std::lround(rd(i * si)) + std::lround(rd(i * si + 1)) * 256 + std::lround(rd(i * si + 2)) * 65536;
        }
    }
    std::unordered_map<int64_t, Transform3D> curr;
    const Array keys = entity_xforms.keys();
    for (int64_t k = 0; k < keys.size(); ++k) curr[static_cast<int64_t>(keys[k])] = entity_xforms[keys[k]];

    // motion + spawned-entity mask: StructuralExporter._compute_motion, line for line
    std::fill(motion_.begin(), motion_.end(), 0.f);
    std::fill(motion_hwc_.begin(), motion_hwc_.end(), 0.f);
    std::fill(no_prev_.begin(), no_prev_.end(), 0.f);
    no_prev_count_ = 0;
    if (prev_valid_) {
        const Projection prev_vp = prev_proj_ * Projection(prev_cam_.affine_inverse());
        const real_t p00 = proj.columns[0][0], p11 = proj.columns[1][1];
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                const size_t pi = static_cast<size_t>(y) * W + x;
                const float d = depth_m_[pi];
                if (std::isinf(d)) continue;
                const int64_t eid = ids_[pi];
                if (eid != 0 && !prev_xforms_.count(eid)) { no_prev_[pi] = 1.f; ++no_prev_count_; continue; }
                // Camera3D.project_position(Vector2(x + 0.5, y + 0.5), d) for this perspective camera
                const real_t nx = ((x + 0.5f) / W) * 2.0f - 1.0f, ny = (1.0f - (y + 0.5f) / H) * 2.0f - 1.0f;
                const Vector3 world_p = cam.xform(Vector3(nx * d / p00, ny * d / p11, -d));
                auto ic = curr.find(eid); auto ip = prev_xforms_.find(eid);
                const Transform3D mt = ic != curr.end() ? ic->second : Transform3D();
                const Transform3D mtm1 = ip != prev_xforms_.end() ? ip->second : Transform3D();
                const Vector3 world_p0 = (mtm1 * mt.affine_inverse()).xform(world_p);
                const Vector4 clip = prev_vp.xform(Vector4(world_p0.x, world_p0.y, world_p0.z, 1.0f));
                if (std::fabs(clip.w) < 1e-9) continue;
                const Vector3 ndc = Vector3(clip.x, clip.y, clip.z) / clip.w;
                const double mu = (ndc.x + 1.0) * 0.5 - (x + 0.5) / static_cast<double>(W);
                const double mv = (1.0 - ndc.y) * 0.5 - (y + 0.5) / static_cast<double>(H);
                motion_[pi] = static_cast<float>(mu); motion_[hw + pi] = static_cast<float>(mv);
                motion_hwc_[pi * 2] = static_cast<float>(mu); motion_hwc_[pi * 2 + 1] = static_cast<float>(mv);
            }
    } else {
        x_prev_ = x_;                                       // masked out entirely by motion_valid = 0
        std::fill(feat_prev_.begin(), feat_prev_.end(), 0.f);
    }
    mvalid_[0] = prev_valid_ ? 1.f : 0.f;
    last_mvalid_ = prev_valid_;

    // bind by NAME, in this engine's tensor order
    structrender::Engine& e = *tengines_[style_];
    std::vector<const float*> in; std::vector<float*> out;
    for (const auto& t : e.tensors()) {
        if (t.is_input) {
            in.push_back(t.name == "x_t" ? x_.data() : t.name == "x_prev" ? x_prev_.data() : t.name == "feat_prev" ? feat_prev_.data()
                         : t.name == "motion" ? motion_.data() : t.name == "no_prev" ? no_prev_.data() : mvalid_.data());
        } else {
            out.push_back(t.name == "rgb" ? rgbf_.data() : feat_.data());
        }
    }
    PackedByteArray rgb;
    if (!e.infer(in, out)) { err_ = String(e.last_error().c_str()); prev_valid_ = false; return rgb; }
    rgb.resize(static_cast<int64_t>(3 * hw));
    uint8_t* o = rgb.ptrw();
    for (size_t i = 0; i < hw; ++i)
        for (int c = 0; c < 3; ++c) {
            float v = rgbf_[c * hw + i];
            if (!std::isfinite(v)) { err_ = "non-finite model output"; prev_valid_ = false; return PackedByteArray(); }
            v = v < 0.f ? 0.f : (v > 1.f ? 1.f : v);
            o[i * 3 + c] = static_cast<uint8_t>(std::lround(v * 255.f));
        }
    // this frame becomes the history of the next
    x_prev_.swap(x_); feat_prev_.swap(feat_);
    prev_cam_ = cam; prev_proj_ = proj; prev_xforms_ = std::move(curr); prev_valid_ = true;
    err_ = "";
    last_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return rgb;
}
