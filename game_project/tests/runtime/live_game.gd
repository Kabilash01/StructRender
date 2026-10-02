## M10 live demo: the PLAYABLE game (main.tscn), rendered live by two-style B,
## with the style switchable while you play (KB Decisions 94, 95).
##
## Per frame: StructuralExporter.capture_live() (the same exporter and camera the
## game dataset was recorded with) -> StructRenderer.render_images() -> shown
## full-screen over the native view. One engine per style, built from the ONE
## two-style checkpoint (scripts/m10_two_style_engines.py); both are loaded once,
## and switching style just picks which renders the next frame.
##
## Keys: WASD / arrows + Space play the game as normal. 1 = oil, 2 = watercolour,
## 0 = native Godot render. If a render fails the native frame shows (§4
## fallback); gameplay never waits on the renderer.
##
## The engines run without frame history (cold start every frame) — see
## eval/reports/game_b_two_style_test_*_nohistory.json for what that costs.
##
## Env: SR_ENGINE_OIL, SR_ENGINE_WATERCOLOR (engine paths, required).
## SR_FRAMES=N runs a scripted check instead of free play: the player walks a
## fixed route, the style cycles oil -> watercolour -> native every 100 frames,
## a frame of each is saved to SR_SAVE_DIR, timing is printed, then it quits.
extends SceneTree

const STYLE_NAMES := ["oil", "watercolour", "native"]

var _style := 0          # 0 oil, 1 watercolour, 2 native
var _prev_keys := {}

func _pressed_once(key: Key) -> bool:
	var down := Input.is_physical_key_pressed(key)
	var was: bool = _prev_keys.get(key, false)
	_prev_keys[key] = down
	return down and not was

func _initialize() -> void:
	DisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_DISABLED)
	Engine.set_meta(&"structrender_no_sim", true)       # the human (or the check) plays
	var scripted := OS.has_environment("SR_FRAMES")
	var frames := int(OS.get_environment("SR_FRAMES")) if scripted else -1

	var main: Node3D = load("res://scenes/main.tscn").instantiate()
	root.add_child(main)
	current_scene = main                                  # the spawner needs it (game_record.gd)
	await process_frame
	var cam: Camera3D = main.get_node("Player").get_node("MainCamera")

	const H := preload("res://tests/structural/structural_harness.gd")
	var adapter := H.make_adapter(self, "live_game")
	var ex := H.make_exporter(self, cam, main, adapter, "live_game")

	var renderers := []
	for env in ["SR_ENGINE_OIL", "SR_ENGINE_WATERCOLOR"]:
		var r := StructRenderer.new()
		if not r.load_engine(OS.get_environment(env)):
			printerr("LIVE_GAME engine ", env, " failed: ", r.last_error(), " — that style shows native")
			r = null
		renderers.append(r)

	var layer := CanvasLayer.new()
	root.add_child(layer)
	var view := TextureRect.new()
	view.set_anchors_preset(Control.PRESET_FULL_RECT)
	view.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	view.stretch_mode = TextureRect.STRETCH_KEEP_ASPECT_CENTERED
	layer.add_child(view)
	var label := Label.new()
	label.position = Vector2(12, 8)
	label.add_theme_font_size_override("font_size", 22)
	layer.add_child(label)
	printerr("LIVE_GAME ready — 1 oil, 2 watercolour, 0 native")

	var t_tot := {0: PackedFloat64Array(), 1: PackedFloat64Array()}
	var fallbacks := 0
	var i := 0
	while frames < 0 or i < frames:
		var t0 := Time.get_ticks_usec()
		if scripted:
			_style = (i / 100) % 3
			var dir := Vector2(sin(i * 0.013), cos(i * 0.009))
			for a in ["move_right", "move_left", "move_back", "move_forward"]:
				Input.action_release(a)
			Input.action_press("move_right" if dir.x > 0 else "move_left", absf(dir.x))
			Input.action_press("move_back" if dir.y > 0 else "move_forward", absf(dir.y))
		else:
			if _pressed_once(KEY_1): _style = 0
			if _pressed_once(KEY_2): _style = 1
			if _pressed_once(KEY_0): _style = 2
		var img: Image = null
		if _style < 2 and renderers[_style] != null:
			var imgs: Array = await ex.capture_live()
			if imgs.size() == 3:
				var rgb: PackedByteArray = renderers[_style].render_images(imgs[0], imgs[1], imgs[2])
				if not rgb.is_empty():
					img = Image.create_from_data(512, 288, false, Image.FORMAT_RGB8, rgb)
			if img == null:
				fallbacks += 1
		else:
			await process_frame
		view.visible = img != null
		if img != null:
			view.texture = ImageTexture.create_from_image(img)
		var dt := (Time.get_ticks_usec() - t0) / 1000.0
		label.text = "%s   %.0f fps   [1 oil  2 watercolour  0 native]" % [STYLE_NAMES[_style], 1000.0 / maxf(dt, 0.001)]
		if scripted:
			if _style < 2 and i % 100 >= 10:
				t_tot[_style].append(dt)
			if OS.has_environment("SR_SAVE_DIR") and i % 100 == 50:
				await process_frame
				root.get_texture().get_image().save_png(
					OS.get_environment("SR_SAVE_DIR") + "/live_%s_%04d.png" % [STYLE_NAMES[_style], i])
		i += 1

	for s in [0, 1]:
		var a: PackedFloat64Array = t_tot[s]
		a.sort()
		if a.size() > 0:
			print("LIVE_GAME %s: frames=%d  median %.1f ms  (%.1f fps)" % [STYLE_NAMES[s], a.size(), a[a.size() / 2], 1000.0 / a[a.size() / 2]])
	print("LIVE_GAME fallbacks=%d" % fallbacks)
	quit(0)
