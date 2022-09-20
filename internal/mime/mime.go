// Package mime defines common mime types.
package mime

import (
	"fmt"

	"google.golang.org/protobuf/types/known/anypb"
)

const (
	// PNG represents a PNG raw leaf node.
	PNG = "image/png"
	// CSSText represents a css raw leaf node (encoded in utf8).
	CSSText = "text/css"
	// HTMLText represents a html raw leaf node (encoded in utf8).
	HTMLText = "text/html"
	// HTMLParent represents a multiscope parent node with a CSS and HTML children node.
	HTMLParent = "multiscope/html"
	// PlainText represents a text raw leaf node (encoded in utf8).
	PlainText = "text/plain"
	// JSON represents a JSON raw leaf node.
	JSON = "application/json"
	// VegaParent represents a parent vega node. Such node has two children:
	// specification and data respectively to stream the specification of the chart
	// and the data of the chart.
	VegaParent = "vega/parent-node"
	// VegaLiteV2 represents a vega lite version 2 JSON raw leaf node.
	VegaLiteV2 = "vega/specification-lite-v2"
	// ScalarTimeSeries is a time series of one or more scalar values.
	ScalarTimeSeries = "data/scalar-time-series"
	// MultiscopeTicker is a ticker to synchronize data.
	MultiscopeTicker = "multiscope/ticker"
	// MultiscopeDMEnvGroup is a group to visualize a dmenv environment.
	MultiscopeDMEnvGroup = "multiscope/dm_env_rpc"
	// MultiscopeTensorGroup is a group to visualize different aspects of a tensor.
	MultiscopeTensorGroup = "multiscope/tensor"
	// MultiscopeRootData is the data of a Multiscope root node.
	MultiscopeRootData = "multiscope/root/data"
	// Protobuf represents a google.protobuf.Any protobuf.
	Protobuf = "application/x-protobuf"
	// Error is for a panel to display a string error.
	Error = "multiscope/error"
	// Unsupported is for a panel to display an unsupported error.
	Unsupported = "multiscope/unsupported"
)

const (
	// NodeNameHTML is the name of the node streaming the HTML body.
	NodeNameHTML = "html"
	// NodeNameCSS is the name of the node streaming CSS.
	NodeNameCSS = "css"
)

// NamedProtobuf extends Protobuf per go/multiscope-rfc #12 to represent a
// protobuf with a specific type, like so:
// application/x-protobuf;proto=google.protobuf.Timestamp.
func NamedProtobuf(name string) string {
	return fmt.Sprintf("%s;proto=%s", Protobuf, name)
}

// AnyToMIME returns the MIME type of an Any proto.
func AnyToMIME(any *anypb.Any) string {
	name := string(any.MessageName())
	return NamedProtobuf(name)
}
