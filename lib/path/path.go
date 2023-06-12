// Package path provide utility function to manipulate node paths in the tree.
package path

import (
	"encoding/base64"

	treepb "multiscope/protos/tree_go_proto"

	"github.com/pkg/errors"
	"google.golang.org/protobuf/proto"
)

// ToBase64 encodes a node path into a base64 string.
func ToBase64(path *treepb.NodePath) (string, error) {
	enc, err := proto.Marshal(path)
	if err != nil {
		return "", errors.Errorf("cannot encode node path to a proto: %v", err)
	}
	return base64.StdEncoding.EncodeToString(enc), nil
}

// FromBase64 decodes a base64 string into a path.
func FromBase64(enc string) (*treepb.NodePath, error) {
	dec, err := base64.StdEncoding.DecodeString(enc)
	if err != nil {
		return nil, errors.Errorf("cannot decode base64 string: %v", err)
	}
	var path treepb.NodePath
	if err := proto.Unmarshal(dec, &path); err != nil {
		return nil, errors.Errorf("cannot unmarshal node path: %v", err)
	}
	return &path, nil
}
