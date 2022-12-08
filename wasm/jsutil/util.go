package jsutil

import "syscall/js"

// Type returns the type of a Javascript object.
// Used for debugging.
func Type(val js.Value) string {
	cstr := val.Get("constructor")
	if cstr.IsNull() {
		return val.Type().String()
	}
	name := cstr.Get("name")
	if name.IsNull() {
		return val.Type().String()
	}
	return name.String()
}
