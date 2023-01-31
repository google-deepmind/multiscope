// Package timeline provides interfaces to serialize nodes in a timeline.
package timeline

import "multiscope/internal/server/core"

type (
	// Adapter provides a timeline Marshaler from a node in the tree.
	Adapter interface {
		// Timeline returns a node to store a timeline.
		Timeline() core.Node
	}
)
