package events

import (
	pb "multiscope/protos/tree_go_proto"
	"strings"
)

// Callback to process events.
type Callback func(*pb.Event) error

// FilterProto returns a callback filtering events only for a given type of protocol buffers.
func (cb Callback) FilterProto(url string) Callback {
	curl := CoreURL(url)
	return func(ev *pb.Event) error {
		if CoreURL(ev.GetPayload().GetTypeUrl()) != curl {
			return nil
		}
		return cb(ev)
	}
}

// CoreURL extracts the core URL of the proto of an event.
func CoreURL(url string) string {
	if !strings.HasPrefix(url, "type.google") {
		return url
	}
	sp := strings.Split(url, "/")
	if len(sp) != 2 {
		return url
	}
	return sp[1]
}
