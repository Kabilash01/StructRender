## M10: render one recorded frame through the StructRenderer GDExtension.
## Inputs are raw float32 dumps of the frame's depth / normal / albedo written by
## scripts/m10_gdext_parity.py; output is the RGB8 bytes, compared in Python.
extends SceneTree

func _initialize() -> void:
	var dir := OS.get_environment("SR_PARITY_DIR")
	var engine_path := OS.get_environment("SR_ENGINE")
	var r := StructRenderer.new()
	if not r.load_engine(engine_path):
		printerr("PARITY load failed: ", r.last_error()); quit(1); return
	var d := FileAccess.get_file_as_bytes(dir + "/depth.bin").to_float32_array()
	var n := FileAccess.get_file_as_bytes(dir + "/normal.bin").to_float32_array()
	var a := FileAccess.get_file_as_bytes(dir + "/albedo.bin").to_float32_array()
	var rgb := r.render(d, n, a, 512, 288)
	if rgb.is_empty():
		printerr("PARITY render failed: ", r.last_error()); quit(1); return
	var f := FileAccess.open(dir + "/rgb8.bin", FileAccess.WRITE)
	f.store_buffer(rgb); f.close()
	# the fallback contract: bad input must come back EMPTY, not crash
	var bad := r.render(PackedFloat32Array(), n, a, 512, 288)
	print("PARITY ok  ms=%.2f  bad_input_empty=%s  err='%s'" % [r.last_ms(), bad.is_empty(), r.last_error()])
	quit(0)
