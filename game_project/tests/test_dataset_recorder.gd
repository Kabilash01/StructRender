extends SceneTree
## Milestone 2 — dataset recorder logic that does NOT need a GPU (runs in the
## headless M0 suite via scripts/test.sh). The GPU-bound checks (buffer content,
## determinism, manifest well-formedness, reappearance semantics) are in
## game_project/tests/dataset/d1_recorder.gd behind scripts/test_dataset.sh.
##
## Covered here:
##   - dataset session-id namespace is `ds_<ts>_<hash>` (distinct from GameLog's
##     `session_<ts>` — KB Decision 22),
##   - immutability: begin() refuses a session that already has a manifest, and
##     refuses one carrying a VALIDATED marker (the validator writes that; the
##     recorder only checks for it),
##   - debug log vs dataset manifest live under different roots, no shared files.

const Harness := preload("res://tests/harness.gd")
const Recorder := preload("res://instrumentation/dataset_recorder.gd")
const Adapter := preload("res://instrumentation/entity_lifecycle_adapter.gd")
const DH := preload("res://tests/dataset/dataset_harness.gd")


func _initialize() -> void:
	var root_dir := ProjectSettings.globalize_path("user://_ds_recorder_test_%d" % Time.get_ticks_usec())
	for _i in 5:
		await process_frame  # let autoloads (GameLog) finish _ready, and 3D nodes enter the world

	# ---- session-id namespace ----
	var sid := DH.new_session_id()
	Harness.check(sid.begins_with("ds_") and not sid.begins_with("session_"),
		"dataset session_id is in the ds_ namespace, not GameLog's session_")
	var rx := RegEx.new()
	rx.compile("^ds_[0-9]+_[0-9a-f]{8}$")
	Harness.check(rx.search(sid) != null, "session_id matches ds_<unix_ts>_<8 hex>: %s" % sid)

	var world := Node3D.new()
	root.add_child(world)
	var cam := Camera3D.new()
	world.add_child(cam)
	var adapter := Adapter.new()
	root.add_child(adapter)
	adapter.session_id = sid

	# ---- immutability: clean start, then refuse to overwrite ----
	var rec := _make_recorder(sid, root_dir, adapter, cam)
	Harness.check(rec.begin(), "begin() clean start on a fresh session")
	var manifest_jsonl := "%s/%s/manifest.jsonl" % [root_dir, sid]
	Harness.check(FileAccess.file_exists(manifest_jsonl), "begin() created manifest.jsonl")
	FileAccess.open(manifest_jsonl, FileAccess.READ_WRITE).store_line("{\"sentinel\": true}")

	var rec_again := _make_recorder(sid, root_dir, adapter, cam)
	Harness.check(not rec_again.begin(), "begin() refuses a session that already has a manifest.jsonl")
	Harness.check(FileAccess.get_file_as_string(manifest_jsonl).contains("sentinel"),
		"refused begin() left the existing manifest untouched")

	# ---- immutability: VALIDATED marker (written by the validator, not us) ----
	var sid_v := DH.new_session_id()
	var vdir := "%s/%s" % [root_dir, sid_v]
	DirAccess.make_dir_recursive_absolute(vdir)
	FileAccess.open("%s/VALIDATED" % vdir, FileAccess.WRITE).store_string("{}")
	var rec_v := _make_recorder(sid_v, root_dir, adapter, cam)
	Harness.check(not rec_v.begin(), "begin() refuses a session carrying a VALIDATED marker")
	Harness.check(not FileAccess.file_exists("%s/manifest.jsonl" % vdir),
		"refused begin() wrote no manifest into the VALIDATED session")

	# ---- debug log vs dataset manifest: separate roots, no shared files ----
	var glog := root.get_node_or_null("GameLog")
	var glog_path := "" if glog == null else ProjectSettings.globalize_path(String(glog.get_log_path()))
	var logs_dir := ProjectSettings.globalize_path("user://logs")
	Harness.check(glog_path != "" and glog_path.begins_with(logs_dir),
		"GameLog writes under user://logs (%s)" % glog_path)
	Harness.check(glog_path.get_file().begins_with("session_"),
		"GameLog file is session_<ts>.jsonl, a different namespace from ds_")
	Harness.check(not root_dir.begins_with(logs_dir), "dataset root is not under user://logs")
	Harness.check(not glog_path.begins_with(root_dir), "GameLog path is not under the dataset root")
	var ds_files := {}
	for f in DirAccess.get_files_at("%s/%s" % [root_dir, sid]):
		ds_files[f] = true
	Harness.check(not ds_files.has(glog_path.get_file()),
		"the GameLog file name does not appear in the dataset session dir")

	_rmtree(root_dir)
	Harness.finish(self)


func _make_recorder(sid: String, root_dir: String, adapter: Node, cam: Camera3D) -> Node:
	var rec := Recorder.new()
	root.add_child(rec)
	var fake_exporter := Node.new()
	rec.add_child(fake_exporter)
	rec.configure(sid, root_dir, fake_exporter, adapter, cam, Vector2i(512, 288), 60)
	return rec


func _rmtree(path: String) -> void:
	var d := DirAccess.open(path)
	if d == null:
		return
	for f in d.get_files():
		DirAccess.remove_absolute("%s/%s" % [path, f])
	for sub in d.get_directories():
		_rmtree("%s/%s" % [path, sub])
	DirAccess.remove_absolute(path)
