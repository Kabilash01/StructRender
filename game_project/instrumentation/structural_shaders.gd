extends RefCounted
## Override-shader source for the Milestone 1 structural exporter.
##
## Each pass replaces every mesh's material with one of these (unshaded, so
## engine lighting/tonemap does not touch the value) and reads the result back
## from a half-float SubViewport. See data_pipeline/schemas/frame_structure.md.
##
## Loaded via `preload("res://instrumentation/structural_shaders.gd")` (no
## class_name — matches the M0 harness pattern for editor-less runs).

## Linear eye-space Z in metres (-view_pos.z), written to R. Background renders
## black against the exporter's pure-black environment => R <= 0 marks "no geometry".
const DEPTH := """
shader_type spatial;
render_mode unshaded, cull_disabled, shadows_disabled, fog_disabled;
varying vec3 v_view_pos;
void vertex() {
	v_view_pos = (MODELVIEW_MATRIX * vec4(VERTEX, 1.0)).xyz;
}
void fragment() {
	ALBEDO = vec3(-v_view_pos.z, 0.0, 0.0);
}
"""

## World-space unit normal, stored RAW in [-1, 1] (half-float target keeps sign).
const NORMAL := """
shader_type spatial;
render_mode unshaded, cull_disabled, shadows_disabled, fog_disabled;
varying vec3 v_world_normal;
void vertex() {
	v_world_normal = MODEL_NORMAL_MATRIX * NORMAL;
}
void fragment() {
	ALBEDO = normalize(v_world_normal);
}
"""

## Flat linear base colour (set per-mesh via the `base_color` uniform).
const ALBEDO := """
shader_type spatial;
render_mode unshaded, cull_disabled, shadows_disabled, fog_disabled;
uniform vec3 base_color = vec3(1.0);
void fragment() {
	ALBEDO = base_color;
}
"""

## instance_id split little-endian across RGB as integer channel values 0..255
## (exact in a half-float target); decoded with round() on readback.
const ENTITY_ID := """
shader_type spatial;
render_mode unshaded, cull_disabled, shadows_disabled, fog_disabled;
uniform float enc_r = 0.0;
uniform float enc_g = 0.0;
uniform float enc_b = 0.0;
void fragment() {
	ALBEDO = vec3(enc_r, enc_g, enc_b);
}
"""


static func make(code: String) -> Shader:
	var s := Shader.new()
	s.code = code
	return s
