package core

import (
	"go.uber.org/multierr"
)

// NodeResetter resets the node.
type NodeResetter interface {
	ResetNode() error
}

func recursiveReset(node Node) error {
	reseter, ok := node.(NodeResetter)
	if ok {
		if err := reseter.ResetNode(); err != nil {
			return err
		}
	}
	parent, ok := node.(Parent)
	if !ok {
		return nil
	}
	children, err := parent.Children()
	if err != nil {
		return err
	}
	var gErr error
	for _, childName := range children {
		child, err := parent.Child(childName)
		if err != nil {
			return err
		}
		if err := recursiveReset(child); err != nil {
			gErr = multierr.Append(gErr, err)
		}
	}
	return gErr
}

// RecursiveReset resets all children nodes implementing the NodeResetter interface.
func RecursiveReset(parent ParentNode, withPath WithPBPath) error {
	_, node, err := pathToNode(parent, withPath)
	if err != nil {
		return err
	}
	return recursiveReset(node)
}
