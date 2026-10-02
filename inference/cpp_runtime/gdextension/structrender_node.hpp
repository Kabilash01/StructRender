// StructRenderer — the Godot-facing half of the M10 live loop (KB §4, §10 M10).
#pragma once
#include <godot_cpp/classes/image.hpp>
#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/packed_byte_array.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>
#include <godot_cpp/variant/projection.hpp>
#include <godot_cpp/variant/transform3d.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/variant/string.hpp>

#include <memory>
#include <unordered_map>
#include <vector>

#include "structrender/engine.hpp"

namespace godot {

class StructRenderer : public RefCounted {
    GDCLASS(StructRenderer, RefCounted)

public:
    StructRenderer();
    ~StructRenderer() override;

    bool load_engine(const String& path);
    bool is_loaded() const;
    // depth [H*W] linear eye-space metres (+inf background), normal/albedo [H*W*3]
    // interleaved — exactly what StructuralExporter.export_frame() returns and
    // the recorder saved. Returns RGB8 [H*W*3], or an EMPTY array on any failure:
    // the caller must then show the native frame (§4 fallback), never stall.
    PackedByteArray render(const PackedFloat32Array& depth, const PackedFloat32Array& normal,
                           const PackedFloat32Array& albedo, int width, int height);
    // LIVE path (Decision 86): decode the exporter's raw pass Images in C++,
    // reproducing StructuralExporter.export_frame()'s per-pixel loop exactly
    // (depth <= 0 -> background: depth +inf, normal and albedo 0).
    PackedByteArray render_images(const Ref<Image>& depth, const Ref<Image>& normal, const Ref<Image>& albedo);
    String last_error() const;
    double last_ms() const;

    // TEMPORAL live path (two-style B, KB Decision 97). One engine per style, all
    // sharing ONE history: the encoder is style-free, so switching style keeps
    // the previous frame's features valid.
    bool load_temporal_engines(const PackedStringArray& paths);
    void set_style(int style);
    int get_style() const;
    void reset_history();
    // Pass images as capture_live() returns them plus the entity-id pass, and the
    // exporter camera's transform / projection and {instance_id: Transform3D} of
    // every registered entity at this frame. Motion and the spawned-entity mask
    // are computed here, line for line as StructuralExporter._compute_motion.
    PackedByteArray render_temporal(const Ref<Image>& depth, const Ref<Image>& normal, const Ref<Image>& albedo,
                                    const Ref<Image>& ids, const Transform3D& cam, const Projection& proj,
                                    const Dictionary& entity_xforms);
    PackedFloat32Array last_motion() const;   // [H*W*2] HWC, for the parity test
    int last_no_prev_pixels() const;
    bool last_motion_valid() const;

protected:
    static void _bind_methods();

private:
    std::unique_ptr<structrender::Engine> engine_;
    std::vector<float> in_, out_;
    PackedByteArray run_from_input(int w, int h);
    String err_;
    double last_ms_ = 0.0;

    // temporal state
    std::vector<std::unique_ptr<structrender::Engine>> tengines_;
    int style_ = 0;
    std::vector<float> x_, x_prev_, feat_, feat_prev_, motion_, motion_hwc_, no_prev_, mvalid_, rgbf_;
    bool prev_valid_ = false;
    Transform3D prev_cam_;
    Projection prev_proj_;
    std::unordered_map<int64_t, Transform3D> prev_xforms_;
    int no_prev_count_ = 0;
    bool last_mvalid_ = false;
    bool decode_structure(const Ref<Image>& depth, const Ref<Image>& normal, const Ref<Image>& albedo,
                          std::vector<float>& x, std::vector<float>& depth_m);
    std::vector<float> depth_m_;
    std::vector<int64_t> ids_;
};

}  // namespace godot
