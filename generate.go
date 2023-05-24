// Copyright 2023 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package multiscope is used to generate all the files require for Multiscope.
package multiscope

// Generate web-assembly client.

//go:generate cp $GOROOT/misc/wasm/wasm_exec.js ./web/res
//go:generate zsh -c "GOOS=js GOARCH=wasm $GOROOT/bin/go build -o web/res/multiscope.wasm wasm/mains/multiscope/main.go && gzip -9 -f web/res/multiscope.wasm"
