## Parity of the LIVE TEMPORAL path against the RECORDING path (KB Decision 97).
##
## The playable game runs in lockstep (one physics step per frame, tree paused
## while passes render — exactly game_record.gd). Each frame, on the SAME
## instant, StructuralExporter.export_frame() computes motion in GDScript (the
## path every training frame went through) and capture_live_temporal() +
## StructRenderer.render_temporal() compute it in C++. Their motion fields are
## compared per frame. Two consecutive frames' structure, motion and C++ output
## are saved to SR_DUMP_DIR so Python can check the C++ frame against the scored
## model (scripts/m10_live_temporal_parity.py).
##
## Env: SR_ENGINE_OIL, SR_ENGINE_WATERCOLOR, SR_DUMP_DIR, SR_FRAMES (default 120).
extends SceneTree

const H := preload("res://tests/structural/structural_harness.gd")

func _save(path: String, bytes: PackedByteArray) -> void:
	var f := FileAccess.open(path, FileAccess.WRITE)
	f.store_buffer(bytes)
	f.close()

func _initialize() -> void:
	DisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_DISABLED)
	Engine.max_fps = 0
	Engine.max_physics_steps_per_frame = 1
	Engine.set_meta(&"structrender_no_sim", true)
	var frames := int(OS.get_environment("SR_FRAMES")) if OS.has_environment("SR_FRAMES") else 120
	var dump := OS.get_environment("SR_DUMP_DIR")

	var main: Node3D = load("res://scenes/main.tscn").instantiate()
	root.add_child(main)
	current_scene = main
	await process_frame
	var cam: Camera3D = main.get_node("Player").get_node("MainCamera")
	var adapter := H.make_adapter(self, "live_parity")
	var ex := H.make_exporter(self, cam, main, adapter, "live_parity")
	for n in [adapter, ex]:
		n.process_mode = Node.PROCESS_MODE_ALWAYS
	var r := StructRenderer.new()
	if not r.load_temporal_engines(PackedStringArray([OS.get_environment("SR_ENGINE_OIL"), OS.get_environment("SR_ENGINE_WATERCOLOR")])):
		printerr("PARITY FAIL engines: ", r.last_error()); quit(1); return
	paused = true

	var worst := 0.0
	var worst_frame := -1
	var valid_mismatch := 0
	var no_prev_total := 0
	var failures := 0
	for i in frames:
		var dir := Vector2(sin(i * 0.05), cos(i * 0.031))
		for a in ["move_right", "move_left", "move_back", "move_forward"]:
			Input.action_release(a)
		Input.action_press("move_right" if dir.x > 0 else "move_left", absf(dir.x))
		Input.action_press("move_back" if dir.y > 0 else "move_forward", absf(dir.y))
		paused = false
		await physics_frame
		paused = true
		await process_frame

		var fs: Dictionary = await ex.export_frame(i)
		var cap: Array = await ex.capture_live_temporal(i)
		r.set_style(i % 2)                       # history must survive style switches
		var rgb: PackedByteArray = r.render_temporal(cap[0], cap[1], cap[2], cap[3], cap[4], cap[5], cap[6])
		if fs.is_empty() or rgb.is_empty():
			failures += 1
			printerr("PARITY frame ", i, " failed: export='", ex.last_error, "' render='", r.last_error(), "'")
			continue
		if bool(fs.motion_valid) != r.last_motion_valid():
			valid_mismatch += 1
		no_prev_total += r.last_no_prev_pixels()
		var mc: PackedFloat32Array = r.last_motion()
		var mg: PackedFloat32Array = fs.motion
		var d := 0.0
		for k in mg.size():
			d = maxf(d, absf(mg[k] - mc[k]))
		if d > worst:
			worst = d; worst_frame = i
		if dump != "" and (i == frames - 2 or i == frames - 1):
			var tag := "f%d" % (i - frames + 2)     # f0 then f1
			_save(dump + "/%s_depth.bin" % tag, (fs.depth as PackedFloat32Array).to_byte_array())
			_save(dump + "/%s_normal.bin" % tag, (fs.normal as PackedFloat32Array).to_byte_array())
			_save(dump + "/%s_albedo.bin" % tag, (fs.albedo_or_material as PackedFloat32Array).to_byte_array())
			_save(dump + "/%s_motion.bin" % tag, mg.to_byte_array())
			_save(dump + "/%s_rgb.bin" % tag, rgb)
			_save(dump + "/%s_style.txt" % tag, str(i % 2).to_utf8_buffer())
	print("PARITY frames=%d failures=%d motion_valid_mismatch=%d no_prev_pixels=%d max|motion C++ - GDScript|=%s (frame %d)" % [
		frames, failures, valid_mismatch, no_prev_total, str(worst), worst_frame])
	quit(0)
