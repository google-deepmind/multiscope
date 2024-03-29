// Copyright 2023 DeepMind Technologies Limited
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

// Package reflect parses Go instances to automatically creates Multiscope lazy writers.
package reflect

import (
	"fmt"
	"reflect"

	"log"
	"multiscope/clients/go/remote"

	"github.com/pkg/errors"
)

type (
	// ParserState represents the state of the parser exploring objects.
	ParserState struct {
		parent  *ParserState
		parents []Parent
		root    remote.Node
	}

	// TargetGetter is a callback to get the value to display.
	TargetGetter func() any

	// Parser explores a specific type of instance to add its element to the Multiscope tree.
	Parser interface {
		// CanParse returns true if val is supported by the parser.
		CanParse(obj any) bool
		// Parse explores the target returned by the TargetGetter.
		// The path to the new writer, or parent of writers, is returned.
		// Such node can be used to customize the tree built by reflect.
		// For instance, if a group node is returned, then additional children can be
		// added for additional specific visualizations.
		Parse(state *ParserState, name string, fObj TargetGetter) (remote.Node, error)
	}

	// Parent is a node-instance pair in the stack while an object is being explored.
	Parent struct {
		node remote.Node
		obj  any
	}
)

// List of parsers from the most specialized to the most general.
var parsers = []Parser{&scalarParser{}, &structParser{}, &sliceStructParser{}}

func subsequentParsers(skipTo Parser) []Parser {
	if skipTo == nil {
		return parsers
	}
	for i, prsr := range parsers {
		if prsr != skipTo {
			continue
		}
		return parsers[i+1:]
	}
	return []Parser{}
}

func findParser(obj any, skipTo Parser) Parser {
	for _, p := range subsequentParsers(skipTo) {
		if p.CanParse(obj) {
			return p
		}
	}
	return nil
}

// RegisterParser registers a new parser.
// Parsers registered the latest take precedence over parsers registered before.
func RegisterParser(prsr Parser) {
	parsers = append([]Parser{prsr}, parsers...)
}

// Node returns the node in the Multiscope tree.
func (p Parent) Node() remote.Node {
	return p.node
}

// Obj returns the Go instance that the node represents.
func (p Parent) Obj() any {
	return p.obj
}

// NewParserState returns a new state to parse structures for Multiscope.
func NewParserState(parent *ParserState, root remote.Node) *ParserState {
	ps := &ParserState{parent: parent, root: root}
	ps.Push(root, nil)
	return ps
}

// Root returns the root of the tree of the parser.
func (ps *ParserState) Root() remote.Node {
	return ps.root
}

// Push a parent to the stack.
func (ps *ParserState) Push(node remote.Node, obj any) {
	ps.parents = append(ps.parents, Parent{node: node, obj: obj})
}

// Pop a parent from the stack
func (ps *ParserState) Pop() {
	ps.parents = ps.parents[:len(ps.parents)-1]
}

// Parent returns the current parent in the state.
func (ps *ParserState) Parent() Parent {
	return ps.parents[len(ps.parents)-1]
}

func set(dst, src any) bool {
	srcType := reflect.TypeOf(src)
	dstVal := reflect.ValueOf(dst).Elem()
	dstType := dstVal.Type()
	if !srcType.ConvertibleTo(dstType) {
		return false
	}
	dstVal.Set(reflect.ValueOf(src).Convert(srcType))
	return true
}

func (ps *ParserState) find(ptr any) bool {
	for i := range ps.parents {
		obj := ps.parents[len(ps.parents)-i-1].Obj()
		if obj == nil {
			continue
		}
		ok := set(ptr, obj)
		if ok {
			return true
		}
	}
	if ps.parent == nil {
		return set(ptr, ps.root)
	}
	return ps.parent.find(ptr)
}

func (ps *ParserState) buildAvailableParent() []string {
	var av []string
	for i := range ps.parents {
		parent := ps.parents[len(ps.parents)-i-1]
		av = append(av, fmt.Sprintf("%q:%T", parent.Node().Path().Last(), parent.Obj()))
	}
	if ps.parent != nil {
		av = append(av, ps.parent.buildAvailableParent()...)
	}
	return av
}

// Find the first instance that can be cast to the object pointed by ptr and set ptr with that instance.
// An error is returned if no parent can be casted.
func (ps *ParserState) Find(ptr any) error {
	kind := reflect.TypeOf(ptr).Kind()
	if kind != reflect.Ptr {
		log.Fatalf("cannot set ptr because it is not a pointer (type: %T kind: %s)", ptr, kind)
	}
	if ok := ps.find(ptr); !ok {
		return fmt.Errorf("cannot find a parent that can be casted to %T. Available parents are: %v", ptr, ps.buildAvailableParent())
	}
	return nil
}

// Parse an object once the parsing process has started.
// Specify a non-nil parser skipTo to skip all the parsers registered before. This is useful
// for a parser to add something in the tree then to call all subsequent parsers for the same object.
func (ps *ParserState) Parse(name string, fObj TargetGetter, skipTo Parser) (remote.Node, error) {
	obj := fObj()
	if obj == nil {
		return nil, nil
	}
	prsr := findParser(obj, skipTo)
	if prsr == nil {
		parentNode := ps.Parent().Node()
		node, err := remote.NewTextWriter(parentNode.Client(), name, parentNode.Path())
		if err != nil {
			return nil, err
		}
		if err := node.Write(fmt.Sprintf("No parser found for type %T.", obj)); err != nil {
			return nil, err
		}
		return node, nil
	}
	node, err := prsr.Parse(ps, name, fObj)
	if err != nil {
		return nil, fmt.Errorf("error while parsing %T with parser %T: %w", obj, prsr, err)
	}
	return node, err
}

func (ps *ParserState) reflect(root remote.Node, name string, obj any, skipTo Parser) (remote.Node, error) {
	if obj == nil {
		return nil, errors.Errorf("cannot explore a nil instance")
	}
	if reflect.TypeOf(obj).Kind() != reflect.Ptr {
		return nil, errors.Errorf("cannot explore type %T because it is not a pointer", obj)
	}
	return ps.Parse(name, func() any {
		return obj
	}, skipTo)
}

// Reflect continue to parse the tree using a different root.
// All parsers before skipTo are skipped to parse obj.
func (ps *ParserState) Reflect(root remote.Node, name string, obj any, skipTo Parser) (remote.Node, error) {
	return NewParserState(ps, root).reflect(root, name, obj, skipTo)
}

// On parses obj to build the writers necessary to monitor its internal data.
func On(root remote.Node, name string, obj any) (remote.Node, error) {
	return NewParserState(nil, root).reflect(root, name, obj, nil)
}
