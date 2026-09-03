extends RefCounted
## Milestone 2 — per-frame structural buffer storage as compressed .npz.
##
## A .npz is a ZIP archive of .npy members (exactly numpy's np.savez_compressed
## on-disk layout), so the M2 validator / training loaders read a frame with
## `d = numpy.load("frame_00042.npz"); depth = d["depth"]`.
##
## Dtypes (see data_pipeline/schemas/dataset_manifest.md):
##   depth / normal / albedo / motion  -> float16 ('<f2')   (depth keeps +inf)
##   entity_id                         -> int32   ('<i4')
## The M1 frame_structure.md value semantics / ranges / conventions are
## UNCHANGED — only the on-disk precision differs from the M1 debug .npy dumps.
##
## Loaded via preload (no class_name), matching npy_writer.gd.

const MEMBER_KEYS := ["depth", "normal", "albedo", "motion", "entity_id"]


static func _npy_header(descr: String, shape: Array) -> PackedByteArray:
	var shape_str := "("
	for i in shape.size():
		shape_str += str(int(shape[i]))
		if shape.size() == 1 or i < shape.size() - 1:
			shape_str += ", "
	shape_str += ")"
	var header := "{'descr': '%s', 'fortran_order': False, 'shape': %s, }" % [descr, shape_str]
	var unpadded := 10 + header.length() + 1
	var pad := (64 - (unpadded % 64)) % 64
	header += " ".repeat(pad) + "\n"
	var out := PackedByteArray()
	out.append(0x93)
	out.append_array("NUMPY".to_ascii_buffer())
	out.append_array(PackedByteArray([1, 0]))  # version 1.0
	var hlen := header.length()
	out.append_array(PackedByteArray([hlen & 0xFF, (hlen >> 8) & 0xFF]))  # LE uint16
	out.append_array(header.to_ascii_buffer())
	return out


static func npy_f16(values: PackedFloat32Array, shape: Array) -> PackedByteArray:
	var out := _npy_header("<f2", shape)
	var body := PackedByteArray()
	body.resize(values.size() * 2)
	for i in values.size():
		body.encode_half(i * 2, values[i])  # IEEE half; +inf / NaN preserved
	out.append_array(body)
	return out


static func npy_i32(values: PackedInt32Array, shape: Array) -> PackedByteArray:
	var out := _npy_header("<i4", shape)
	out.append_array(values.to_byte_array())
	return out


## fs is the M1 FrameStructure dict. Returns {member_key: npy blob}.
static func frame_members(fs: Dictionary, h: int, w: int) -> Dictionary:
	return {
		"depth": npy_f16(fs.depth, [h, w, 1]),
		"normal": npy_f16(fs.normal, [h, w, 3]),
		"albedo": npy_f16(fs.albedo_or_material, [h, w, 3]),
		"motion": npy_f16(fs.motion, [h, w, 2]),
		"entity_id": npy_i32(fs.entity_id_map, [h, w]),
	}


static func write_npz(path: String, members: Dictionary) -> void:
	var zp := ZIPPacker.new()
	var err := zp.open(path)
	if err != OK:
		push_error("NpzWriter: cannot open %s (err %d)" % [path, err])
		assert(false, "NpzWriter: cannot open output")
		return
	if zp.has_method("set_compression_level"):
		zp.set_compression_level(6)  # deflate
	for key in members:
		zp.start_file("%s.npy" % key)
		zp.write_file(members[key])
		zp.close_file()
	zp.close()
