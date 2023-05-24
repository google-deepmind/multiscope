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

// Package testing provides helper functions for tests.
package testing

import (
	"fmt"
	"math"
	pb "multiscope/protos/tree_go_proto"

	"github.com/google/go-cmp/cmp"
	"go.uber.org/multierr"
)

// CheckMIME returns an error if the MIME type does not match.
func CheckMIME(got, want string) error {
	if diff := cmp.Diff(got, want); len(diff) > 0 {
		return fmt.Errorf("mime type error: got %q but want %q\n%s", got, want, diff)
	}
	return nil
}

// CheckNodePaths returns an error if one or more node path has an error.
func CheckNodePaths(nodes []*pb.Node) error {
	var err error
	for _, node := range nodes {
		if node.Error != "" {
			err = multierr.Append(err, fmt.Errorf("error set for node path %v: %s", node.Path.Path, node.Error))
		}
	}
	return err
}

// Tolerance returns a cmp options to compare vectors approximately equal.
func Tolerance(epsilon float32) cmp.Option {
	return cmp.Comparer(func(x, y float32) bool {
		return float32(math.Abs(float64(x-y))) < epsilon
	})
}
