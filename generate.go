// Package multiscope is used to generate all the files require for Multiscope.
package multiscope

//#go:generate cp $GOROOT/misc/wasm/wasm_exec.js ./web/res

// Generate web-assembly client.

//go:generate zsh -c "GOOS=js GOARCH=wasm go build -o web/res/multiscope.wasm wasm/mains/multiscope/main.go"
