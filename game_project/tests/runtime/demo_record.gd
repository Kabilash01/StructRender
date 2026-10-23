## Demo recording: the playable game, native vs live StructRender, frame by frame.
##
## Lockstep, as game_record.gd and live_temporal_parity.gd: one physics step per
## frame and the tree paused while passes render, so recording cannot slow or
## distort the game, and every saved frame is exactly what the live renderer
## (StructRenderer.render_temporal, history fed back) produced for that instant.
## Live speed is measured separately (KB Decision 98: ~44 fps); this script is
## for a watchable record, so the output plays at 60 fps = real game time.
##
## Per frame it saves <dir>/native_NNNN.png (a native viewport at the dataset
## resolution, set up as dataset_recorder.gd does) and <dir>/styled_NNNN.png plus
## <dir>/style_NNNN.txt. scripts/make_demo_video.py composes them side by side.
##
## Env: SR_ENGINE_OIL, SR_ENGINE_WATERCOLOR, SR_DEMO_DIR,
##      SR_DEMO_SEQ (default "0:300,1:300,0:180,1:180" — style:frames, 0 oil, 1 watercolour).
extends SceneTree

const H := preload("res://tests/structural/structural_harness.gd")

func _initialize() -> void:
	DisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_DISABLED)
	Engine.max_fps = 0
	Engine.max_physics_steps_per_frame = 1
	Engine.set_meta(&"structrender_no_sim", true)
	var out := OS.get_environment("SR_DEMO_DIR")
	var seq_s := OS.get_environment("SR_DEMO_SEQ") if OS.has_environment("SR_DEMO_SEQ") else "0:300,1:300,0:180,1:180"
	var seq: Array = []
	for part in seq_s.split(","):
		var kv := part.split(":")
		for k in int(kv[1]):
			seq.append(int(kv[0]))

	var main: Node3D = load("res://scenes/main.tscn").instantiate()
	root.add_child(main)
	current_scene = main
	await process_frame
	var cam: Camera3D = main.get_node("Player").get_node("MainCamera")
	var adapter := H.make_adapter(self, "demo")
	var ex := H.make_exporter(self, cam, main, adapter, "demo")
	for n in [adapter, ex]:
		n.process_mode = Node.PROCESS_MODE_ALWAYS
	var r := StructRenderer.new()
	if not r.load_temporal_engines(PackedStringArray([OS.get_environment("SR_ENGINE_OIL"), OS.get_environment("SR_ENGINE_WATERCOLOR")])):
		printerr("DEMO FAIL engines: ", r.last_error()); quit(1); return

	# native viewport: dataset_recorder.gd::_setup_native_viewport, same settings
	var vp := SubViewport.new()
	vp.size = Vector2i(512, 288)
	vp.msaa_3d = Viewport.MSAA_DISABLED
	vp.screen_space_aa = Viewport.SCREEN_SPACE_AA_DISABLED
	vp.use_taa = false
	vp.use_hdr_2d = false
	vp.positional_shadow_atlas_size = 0
	vp.world_3d = cam.get_world_3d()
	vp.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	vp.process_mode = Node.PROCESS_MODE_ALWAYS
	root.add_child(vp)
	var ncam := Camera3D.new()
	vp.add_child(ncam)
	paused = true

	var failures := 0
	for i in seq.size():
		var dir := Vector2(sin(i * 0.011), cos(i * 0.007))
		for a in ["move_right", "move_left", "move_back", "move_forward"]:
			Input.action_release(a)
		Input.action_press("move_right" if dir.x > 0 else "move_left", absf(dir.x))
		Input.action_press("move_back" if dir.y > 0 else "move_forward", absf(dir.y))
		paused = false
		await physics_frame
		paused = true
		await process_frame
		# native first, while every material is the game's own
		ncam.global_transform = cam.global_transform
		ncam.fov = cam.fov
		ncam.near = cam.near
		ncam.far = cam.far
		ncam.keep_aspect = cam.keep_aspect
		ncam.current = true
		await process_frame
		await process_frame
		await RenderingServer.frame_post_draw
		vp.get_texture().get_image().save_png(out + "/native_%04d.png" % i)
		r.set_style(seq[i])
		var c: Array = await ex.capture_live_temporal(i)
		var rgb: PackedByteArray = r.render_temporal(c[0], c[1], c[2], c[3], c[4], c[5], c[6])
		if rgb.is_empty():
			failures += 1
			continue
		Image.create_from_data(512, 288, false, Image.FORMAT_RGB8, rgb).save_png(out + "/styled_%04d.png" % i)
		var f := FileAccess.open(out + "/style_%04d.txt" % i, FileAccess.WRITE)
		f.store_string(str(seq[i]))
		f.close()
	print("DEMO frames=%d failures=%d" % [seq.size(), failures])
	quit(0)
