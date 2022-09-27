package settings

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/pkg/errors"
	"google.golang.org/protobuf/proto"
)

type (
	// OnChangeListener is called when a setting is updated.
	OnChangeListener func(src any) error

	// Listener is called when settings change.
	Listener struct {
		// Destination receives the latest value of a setting.
		Destination any
		// Callback called when a new setting value has been
		// unmarshal into Dst.
		Callback OnChangeListener
	}

	// Settings defines an abstract settings interface.
	Settings interface {
		// Listen registers a listener to call when a setting value changes.
		Listen(key string, dst any, f OnChangeListener)

		// Set a value for a key.
		// Dispatch the new value to all the listeners.
		Set(src any, key string, val any)
	}

	// Storage to store the settings.
	Storage interface {
		// Delete a key in the settings.
		Delete(key string) error

		// Load returns the value of a given key.
		Load(key string) string

		// Store a new value for a key.
		Store(key string, buf []byte) error
	}
)

const (
	protoPrefix = "proto"
	jsonPrefix  = "json"
)

func unmarshal(val string, dst any) error {
	before, after, found := strings.Cut(val, ":")
	if !found {
		return errors.Errorf("cannot find setting separator ':'")
	}
	switch before {
	case jsonPrefix:
		return json.Unmarshal([]byte(after), dst)
	case protoPrefix:
		dec, err := base64.StdEncoding.DecodeString(after)
		if err != nil {
			return fmt.Errorf("cannot decode proto buffer: %w", err)
		}
		return proto.Unmarshal(dec, dst.(proto.Message))
	}
	return errors.Errorf("unknown value prefix: %q", before)
}

func updateListener(src any, key, buf string, l Listener) error {
	if err := unmarshal(buf, l.Destination); err != nil {
		return fmt.Errorf("cannot get setting %q=%s: %w", key, buf, err)
	}
	if err := l.Callback(src); err != nil {
		return fmt.Errorf("listener error: %v", err)
	}
	return nil
}

func withPrefix(prefix string, msg []byte) []byte {
	buf := []byte(prefix + ":")
	buf = append(buf, msg...)
	return buf
}

func marshal(val any) ([]byte, error) {
	msg, ok := val.(proto.Message)
	if !ok {
		jbuf, err := json.Marshal(val)
		if err != nil {
			return nil, err
		}
		return withPrefix(jsonPrefix, jbuf), nil
	}
	buf, err := proto.Marshal(msg)
	if err != nil {
		return nil, err
	}
	enc := base64.StdEncoding.EncodeToString(buf)
	return withPrefix(protoPrefix, []byte(enc)), nil
}
