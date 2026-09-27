#include <gdextension_interface.h>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/defs.hpp>
#include <godot_cpp/godot.hpp>

#include "structrender_node.hpp"

using namespace godot;

static void init_module(ModuleInitializationLevel level) {
    if (level == MODULE_INITIALIZATION_LEVEL_SCENE) GDREGISTER_CLASS(StructRenderer);
}
static void uninit_module(ModuleInitializationLevel) {}

extern "C" GDExtensionBool GDE_EXPORT structrender_init(GDExtensionInterfaceGetProcAddress get_proc,
                                                         GDExtensionClassLibraryPtr lib,
                                                         GDExtensionInitialization* init) {
    GDExtensionBinding::InitObject obj(get_proc, lib, init);
    obj.register_initializer(init_module);
    obj.register_terminator(uninit_module);
    obj.set_minimum_library_initialization_level(MODULE_INITIALIZATION_LEVEL_SCENE);
    return obj.init();
}
