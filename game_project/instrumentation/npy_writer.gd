extends RefCounted
## Minimal NumPy .npy v1.0 writer (no numpy on the Godot side; consumers read
## these with numpy at Milestone 2 and with scripts/visualize_frame.py now).
##
## Only the two dtypes M1 needs: little-endian float32 ('<f4') and int32 ('<i4').
## Loaded via preload (no class_name).


static func save_float32(path: String, values: PackedFloat32Array, shape: Array) -> void:
	_write(path, values.to_byte_array(), "<f4", shape)


static func save_int32(path: String, values: PackedInt32Array, shape: Array) -> void:
	_write(path, values.to_byte_array(), "<i4", shape)


static func _write(path: String, raw: PackedByteArray, descr: String, shape: Array) -> void:
	var shape_str := "("
	for i in shape.size():
		shape_str += str(int(shape[i]))
		if shape.size() == 1 or i < shape.size() - 1:
			shape_str += ", "
	shape_str += ")"

	var header := "{'descr': '%s', 'fortran_order': False, 'shape': %s, }" % [descr, shape_str]
	# total header length (10 preamble bytes + header + newline) padded to 64.
	var unpadded := 10 + header.length() + 1
	var pad := (64 - (unpadded % 64)) % 64
	header += " ".repeat(pad) + "\n"

	var out := PackedByteArray()
	out.append_array(PackedByteArray([0x93]))
	out.append_array("NUMPY".to_ascii_buffer())
	out.append_array(PackedByteArray([1, 0]))  # version 1.0
	var hlen := header.length()
	out.append_array(PackedByteArray([hlen & 0xFF, (hlen >> 8) & 0xFF]))  # little-endian uint16
	out.append_array(header.to_ascii_buffer())
	out.append_array(raw)

	var f := FileAccess.open(path, FileAccess.WRITE)
	if f == null:
		push_error("NpyWriter: cannot open %s (err %d)" % [path, FileAccess.get_open_error()])
		assert(false, "NpyWriter: cannot open output")
		return
	f.store_buffer(out)
	f.close()
