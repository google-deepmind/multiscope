package remote

import (
	"multiscope/internal/server/events"
	eventspb "multiscope/protos/events_go_proto"
	treepb "multiscope/protos/tree_go_proto"

	"google.golang.org/protobuf/proto"
)

// KeyboardCallback defines the function type to receive keyboard events from the frontend.
type KeyboardCallback func(ke *eventspb.Keyboard) error

var keyboardProtoURL = proto.MessageName(&eventspb.Keyboard{})

// RegisterKeyboardCallback registers a callback for a given path. If the path is nil, the callback will be called for all paths.
func RegisterKeyboardCallback(clt *Client, kc KeyboardCallback) error {
	cb := events.Callback(func(ev *treepb.Event) error {
		ke := &eventspb.Keyboard{}
		if err := ev.GetPayload().UnmarshalTo(ke); err != nil {
			return err
		}
		return kc(ke)
	}).FilterProto(keyboardProtoURL)
	clt.EventsManager().NewQueue(cb)
	return clt.Display().SetCapture(true)
}
