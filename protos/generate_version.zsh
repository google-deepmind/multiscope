#!/bin/zsh

VERSION=`date '+%Y%m%d_%H%M%S'`

CONTENT=$(cat <<EOF
package protos

// Version of the proto API.
// A new version is generated every time the protocol buffers are
// generated with go generate.
const Version = "$VERSION"

EOF
)

echo $CONTENT > version.go
