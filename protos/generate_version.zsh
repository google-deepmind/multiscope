#!/bin/zsh

VERSION=`date '+%Y%m%d_%H%M%S'`

CONTENT=$(cat <<EOF
// Package version specifies the gRPC API version.
package version

// Version of the proto API.
// A new version is generated every time the protocol buffers are
// generated with go generate.
const Version = "$VERSION"

EOF
)

echo $CONTENT > ../internal/version/version.go
