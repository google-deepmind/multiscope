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

// Package writers provides the gRPC registry registering all the server writers.
package writers

import (
	"multiscope/internal/server/root"
	"multiscope/internal/server/treeservice"
	"multiscope/internal/server/writers/base"
	"multiscope/internal/server/writers/scalar"
	"multiscope/internal/server/writers/tensor"
	"multiscope/internal/server/writers/text"
	"multiscope/internal/server/writers/ticker"
)

// All returns all the writers available by default.
func All() []treeservice.RegisterServiceCallback {
	return []treeservice.RegisterServiceCallback{
		base.RegisterService,
		root.RegisterService,
		scalar.RegisterService,
		text.RegisterService,
		ticker.RegisterService,
		tensor.RegisterService,
	}
}
