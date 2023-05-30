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

package text

import (
	"multiscope/internal/mime"
	"multiscope/internal/server/core"
	"multiscope/internal/server/treeservice"
	"multiscope/internal/server/writers/base"
	treepb "multiscope/protos/tree_go_proto"
)

// HTMLWriter represents a formatted HTML rubric in the web page.
type HTMLWriter struct {
	*base.Group
	html     *base.RawWriter
	css      *base.RawWriter
	htmlData string
	cssData  string
}

var _ core.Node = (*HTMLWriter)(nil)

// NewHTMLWriter returns a new writer to display text.
func NewHTMLWriter() *HTMLWriter {
	w := &HTMLWriter{
		Group: base.NewGroup(mime.HTMLParent),
		html:  base.NewRawWriter(mime.HTMLText),
		css:   base.NewRawWriter(mime.CSSText),
	}
	w.Group.AddChild(mime.NodeNameHTML, w.html)
	w.Group.AddChild(mime.NodeNameCSS, w.css)
	return w
}

// AddToTree adds the writer to the Multiscope tree.
// Also forwards the activity from the data and specification children nodes to the vega nodes
// such that the writer is also active when one of the child is active.
func (w *HTMLWriter) AddToTree(state *treeservice.State, path *treepb.NodePath) (*core.Path, error) {
	writerPath, err := core.SetNodeAt(state.Root(), path, w)
	if err != nil {
		return nil, err
	}
	w.SetForwards(state, writerPath)
	return writerPath, nil
}

// SetForwards sets up the forward from the writer children nodes to the parent.
func (w *HTMLWriter) SetForwards(state *treeservice.State, writerPath *core.Path) {
	state.PathLog().Forward(writerPath, writerPath.PathTo(mime.NodeNameHTML))
	state.PathLog().Forward(writerPath, writerPath.PathTo(mime.NodeNameCSS))
}

// WriteCSS writes the CSS to use with the HTML.
func (w *HTMLWriter) WriteCSS(text string) error {
	w.cssData = text
	err := w.css.Write([]byte(text))
	return err
}

// Write writes HTML as utf-8 encoded text, which might be sanitized when being displayed.
func (w *HTMLWriter) Write(text string) error {
	w.htmlData = text
	return w.html.Write([]byte(text))
}

// HTML returns the HTML string value written by the writer.
func (w *HTMLWriter) HTML() string {
	return w.htmlData
}

// CSS returns the CSS string value written by the writer.
func (w *HTMLWriter) CSS() string {
	return w.cssData
}
