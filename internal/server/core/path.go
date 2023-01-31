package core

import (
	"fmt"
	"strings"

	pb "multiscope/protos/tree_go_proto"
)

// ChildrenNames returns the list of children of a parent as a string.
// Often used for error messages.
func ChildrenNames(parent Parent) string {
	children, err := parent.Children()
	if err != nil {
		return fmt.Sprintf("list of children unavailable: %v", err)
	}
	return fmt.Sprintf("%v", children)
}

func str(path []string) string {
	return strings.Join(path, "/")
}

// SetNodeAt sets a node in the tree given a path.
func SetNodeAt(root Root, pbPath *pb.NodePath, node Node) (*Path, error) {
	if pbPath == nil {
		return nil, fmt.Errorf("cannot set a node at a path set to nil")
	}
	path := pbPath.GetPath()
	if len(path) == 0 {
		return nil, fmt.Errorf("cannot set a node at an empty path")
	}
	name := path[len(path)-1]
	parentPath := path[0 : len(path)-1]
	return root.Path().PathTo(parentPath...).AddChild(name, node)
}

// PathToNode returns the node given a path or an error if the node cannot be found.
func PathToNode(root Node, path []string) (current Node, err error) {
	current = root
	for i, name := range path {
		parent, ok := current.(Parent)
		if !ok {
			return nil, fmt.Errorf("node %s (%T) is not a parent node: cannot get child %s", str(path[:i]), current, path)
		}
		current, err = parent.Child(name)
		if err != nil {
			return nil, fmt.Errorf("error getting child '%v' from path %v: %v", name, str(path[:i]), err)
		}
		if current == nil {
			return nil, fmt.Errorf("%q is not a child of parent node %v (available children are: %v)", name, str(path[:i]), ChildrenNames(parent))
		}
	}
	return
}

// Key is a path represented as a string.
type Key string

// PathSeparator is the separator used in stream's paths.
const PathSeparator = "/"

// ToKey returns a path as a string Key.
func ToKey(s []string) Key {
	protected := make([]string, len(s))
	for i, si := range s {
		si = strings.ReplaceAll(si, `\`, `\\`)
		si = strings.ReplaceAll(si, PathSeparator, `\`+PathSeparator)
		protected[i] = si
	}
	return Key(strings.Join(protected, PathSeparator))
}

func hasProtectedSuffix(s string) bool {
	n := 0
	for strings.HasSuffix(s, `\`) {
		n++
		s = s[:len(s)-1]
	}
	return n%2 > 0
}

// Split a key into its different path elements.
func (k Key) Split() []string {
	r := strings.Split(string(k), "/")
	pr := []string{}
	for _, ri := range r {
		last := ""
		if len(pr) > 0 {
			last = pr[len(pr)-1]
		}
		if hasProtectedSuffix(last) {
			pr[len(pr)-1] = last[:len(last)-1] + "/" + ri
		} else {
			pr = append(pr, ri)
		}
	}
	for i, pi := range pr {
		pr[i] = strings.ReplaceAll(pi, `\\`, `\`)
	}
	return pr
}

// Path represents a path in the stream tree.
type Path struct {
	root Node
	path []string
}

// NewPath returns a new path given a root node.
func NewPath(root Node) *Path {
	return &Path{root: root}
}

// Root returns the root of the path.
func (p *Path) Root() Node {
	return p.root
}

// Node returns the node pointed by the path.
func (p *Path) Node() (Node, error) {
	return PathToNode(p.root, p.path)
}

// AddChild add a child to a parent given a path.
func (p *Path) AddChild(name string, child Node) (*Path, error) {
	node, err := p.Node()
	if err != nil {
		return nil, err
	}
	adder, ok := node.(ChildAdder)
	if !ok {
		return nil, fmt.Errorf("cannot add child %s to path %s: cannot cast %T to core.ChildAdder", name, p, node)
	}
	childName := adder.AddChild(name, child)
	return &Path{
		root: p.root,
		path: append(append([]string{}, p.path...), childName),
	}, nil
}

// PathTo returns a new Path by appending the given path elements to the current Path.
func (p *Path) PathTo(path ...string) *Path {
	return &Path{root: p.root, path: append(append([]string{}, p.path...), path...)}
}

// PB returns the Path as the stream protocol buffer path.
func (p *Path) PB() *pb.NodePath {
	return &pb.NodePath{
		Path: p.path,
	}
}

// ToKey returns the path as a key (typically used to store the path in a map as a key).
func (p *Path) ToKey() Key {
	return ToKey(p.path)
}

// Path returns the path from the root.
func (p *Path) Path() []string {
	return append([]string{}, p.path...)
}

// String returns the path as a string.
func (p *Path) String() string {
	return str(p.path)
}
