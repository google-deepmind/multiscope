// Package wasmurl provides the URL for the Multiscope WASM file.
package wasmurl

var url = "/res/multiscope.wasm"

// Set the URL for the WASM file.
func Set(u string) {
	url = u
}

// URL returns the WASM url.
func URL() string {
	return url
}
