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

// Package jsutil provides helper functions for javascript.
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
