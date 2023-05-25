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

package events

import (
	pb "multiscope/protos/tree_go_proto"
	"strings"
)

// Callback to process events.
type Callback func(*pb.Event) error

// FilterProto returns a callback filtering events only for a given type of protocol buffers.
func (cb Callback) FilterProto(url string) Callback {
	curl := CoreURL(url)
	return func(ev *pb.Event) error {
		if CoreURL(ev.GetPayload().GetTypeUrl()) != curl {
			return nil
		}
		return cb(ev)
	}
}

// CoreURL extracts the core URL of the proto of an event.
func CoreURL(url string) string {
	if !strings.HasPrefix(url, "type.google") {
		return url
	}
	sp := strings.Split(url, "/")
	if len(sp) != 2 {
		return url
	}
	return sp[1]
}
