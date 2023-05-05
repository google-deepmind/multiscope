// Package multiscope is used to generate all the files require for Multiscope.
package multiscope

// Generate web-assembly client.

//go:generate cp $GOROOT/misc/wasm/wasm_exec.js ./web/res
//go:generate zsh -c "GOOS=js GOARCH=wasm go build -o web/res/multiscope.wasm wasm/mains/multiscope/main.go && gzip -9 -f web/res/multiscope.wasm"
