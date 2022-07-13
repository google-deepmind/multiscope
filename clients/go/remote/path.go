package remote

import (
	pb "multiscope/protos/tree_go_proto"
)

type pathGetter interface {
	GetPath() *pb.NodePath
}

// Path represents a path in the tree.
type Path []string

func toPath(p pathGetter) Path {
	return p.GetPath().Path
}

// HasPrefix return true if the receiver starts with prefix.
func (p Path) HasPrefix(prefix Path) bool {
	if len(p) < len(prefix) {
		return false
	}
	for i, si := range prefix {
		if p[i] != si {
			return false
		}
	}
	return true
}

// Append builds a path to a child. If path is nil, then the child will be at the root.
func (p Path) Append(name ...string) Path {
	toChild := append([]string{}, p...)
	return append(toChild, name...)
}

// Last returns the last element of the path.
func (p Path) Last() string {
	if len(p) == 0 {
		return ""
	}
	return p[len(p)-1]
}

// NodePath converts the path to a NodePath protocol buffer.
func (p Path) NodePath() *pb.NodePath {
	return &pb.NodePath{Path: p}
}
