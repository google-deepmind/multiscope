// Package testing provides helper functions for tests.
package testing

import (
	"fmt"
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
