// StructRenderer — the Godot-facing half of the M10 live loop (KB §4, §10 M10).
#pragma once
#include <godot_cpp/classes/image.hpp>
#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/packed_byte_array.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/variant/string.hpp>

#include <memory>
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

protected:
    static void _bind_methods();

private:
    std::unique_ptr<structrender::Engine> engine_;
    std::vector<float> in_, out_;
    PackedByteArray run_from_input(int w, int h);
    String err_;
    double last_ms_ = 0.0;
};

}  // namespace godot
