## Decision 86: the LIVE path (capture_live + render_images, C++ decode) must give
## byte-identical output to the RECORDING path (export_frame + render, GDScript
## decode) on the same frame — i.e. the live renderer sees exactly the input
## distribution the dataset was recorded with.
extends SceneTree

const H := preload("res://tests/structural/structural_harness.gd")
const DH := preload("res://tests/dataset/dataset_harness.gd")

func _initialize() -> void:
	DisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_DISABLED)
	var sc: Dictionary = await DH.build_scene(self, true)
	await process_frame
	var ex := H.make_exporter(self, sc.cam, sc.world, H.make_adapter(self, "eq"), "eq")
	var r := StructRenderer.new()
	if not r.load_engine(OS.get_environment("SR_ENGINE")):
		printerr("EQUIV load failed: ", r.last_error()); quit(1); return
	var worst := 0
	for i in 3:
		var fs: Dictionary = await ex.export_frame(i)
		var a := r.render(fs["depth"], fs["normal"], fs["albedo_or_material"], 512, 288)
		var imgs: Array = await ex.capture_live()
		printerr("EQUIV formats: ", imgs[0].get_format(), " ", imgs[1].get_format(), " ", imgs[2].get_format(), "  RGBAH=", Image.FORMAT_RGBAH, " RGBAF=", Image.FORMAT_RGBAF, " RGBH=", Image.FORMAT_RGBH, " RGBF=", Image.FORMAT_RGBF)
		var b := r.render_images(imgs[0], imgs[1], imgs[2])
		if a.is_empty() or b.is_empty():
			printerr("EQUIV render failed: ", r.last_error()); quit(1); return
		var diff := 0
		for k in a.size():
			diff = max(diff, absi(int(a[k]) - int(b[k])))
		worst = max(worst, diff)
		printerr("EQUIV frame %d: max byte diff %d  (format %d)" % [i, diff, imgs[0].get_format()])
	printerr("EQUIV %s worst=%d" % ["PASS" if worst == 0 else "FAIL", worst])
	quit(0 if worst == 0 else 1)
