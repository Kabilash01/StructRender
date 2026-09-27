## M10 live loop: the training scene, rendered frame by frame through StructRenderer.
##
## Per frame: advance the scene -> StructuralExporter.export_frame() (the SAME
## exporter and scene the dataset was recorded with, so the model sees what it was
## trained on) -> StructRenderer.render() -> display. If render returns empty, the
## frame shows the native Godot render instead (§4 fallback): gameplay never waits
## on the renderer. Prints per-stage timing; quits after SR_FRAMES frames.
extends SceneTree

const H := preload("res://tests/structural/structural_harness.gd")
const DH := preload("res://tests/dataset/dataset_harness.gd")

func _initialize() -> void:
	DisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_DISABLED)
	var frames := int(OS.get_environment("SR_FRAMES")) if OS.has_environment("SR_FRAMES") else 300
	var sc: Dictionary = await DH.build_scene(self, true)
	await process_frame
	var adapter := H.make_adapter(self, "live_session")
	var ex := H.make_exporter(self, sc.cam, sc.world, adapter, "live_session")
	var r := StructRenderer.new()
	var loaded := r.load_engine(OS.get_environment("SR_ENGINE"))
	if not loaded:
		printerr("LIVE engine load failed: ", r.last_error(), " — running native-only (fallback)")
	var view := TextureRect.new()
	view.set_anchors_preset(Control.PRESET_FULL_RECT)
	view.stretch_mode = TextureRect.STRETCH_KEEP_ASPECT_CENTERED
	root.add_child(view)
	var cam_home: Transform3D = sc.cam.global_transform
	var t_exp := PackedFloat64Array(); var t_ren := PackedFloat64Array(); var t_tot := PackedFloat64Array()
	var fallbacks := 0
	for i in frames:
		var t0 := Time.get_ticks_usec()
		# a slow camera sweep so the view actually changes
		var yaw := sin(i * 0.02) * 25.0
		sc.cam.global_transform = Transform3D(Basis(Vector3.UP, deg_to_rad(yaw)) * cam_home.basis, cam_home.origin)
		await process_frame
		var t1 := Time.get_ticks_usec()
		var fs: Dictionary = ex.export_frame(i)
		var t2 := Time.get_ticks_usec()
		var img: Image = null
		if loaded and not fs.is_empty():
			var rgb: PackedByteArray = r.render(fs["depth"], fs["normal"], fs["albedo_or_material"], 512, 288)
			if not rgb.is_empty():
				img = Image.create_from_data(512, 288, false, Image.FORMAT_RGB8, rgb)
		if img == null:
			fallbacks += 1          # native frame stays on screen; the loop never stalls
		else:
			view.texture = ImageTexture.create_from_image(img)
		var t3 := Time.get_ticks_usec()
		if i >= 10:
			t_exp.append((t2 - t1) / 1000.0); t_ren.append((t3 - t2) / 1000.0); t_tot.append((t3 - t0) / 1000.0)
	t_exp.sort(); t_ren.sort(); t_tot.sort()
	var mid := func(a: PackedFloat64Array) -> float: return a[a.size() / 2]
	print("LIVE frames=%d fallbacks=%d  export %.1f ms  render %.1f ms  total %.1f ms  (%.1f fps median)" % [
		frames, fallbacks, mid.call(t_exp), mid.call(t_ren), mid.call(t_tot), 1000.0 / mid.call(t_tot)])
	quit(0)
