// Package settings stores user settings.
package settings

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"
	"syscall/js"

	"github.com/pkg/errors"
	"google.golang.org/protobuf/proto"
)

const prefix = "multiscope/"

// Settings stores settings in cookies.
type Settings struct {
	fErr    func(error)
	storage js.Value
}

func NewSettings(fErr func(error)) *Settings {
	s := &Settings{
		fErr:    fErr,
		storage: js.Global().Get("localStorage"),
	}

	js.Global().Set("multiscopeSettings", js.FuncOf(func(js.Value, []js.Value) interface{} {
		go s.printSettings()
		return nil
	}))

	return s
}

func (s *Settings) printSettings() {
	keys := js.Global().Get("Object").Call("keys", s.storage)
	for i := 0; i < keys.Length(); i++ {
		key := keys.Index(i).String()
		before, after, found := strings.Cut(key, "/")
		if !found || before != prefix[:len(prefix)-1] {
			continue
		}
		fmt.Printf("%q: %v\n", after, s.storage.Get(key).String())
	}
}

func (s *Settings) del(key string) {
	s.storage.Call("removeItem", prefix+key)
}

func (s *Settings) store(key string, buf []byte) {
	s.storage.Set(prefix+key, string(buf))
}

func (s *Settings) load(key string) string {
	val := s.storage.Get(prefix + key)
	if val.IsNull() || val.IsUndefined() {
		return ""
	}
	return val.String()
}

const (
	protoPrefix = "proto"
	jsonPrefix  = "json"
)

func withPrefix(prefix string, msg []byte) []byte {
	buf := []byte(prefix + ":")
	buf = append(buf, msg...)
	return buf
}

func marshal(val interface{}) ([]byte, error) {
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

// Set a key,value pair in the settings.
func (s *Settings) Set(key string, val interface{}) {
	if val == nil {
		s.del(key)
		return
	}
	buf, err := marshal(val)
	if err != nil {
		s.fErr(fmt.Errorf("cannot store key %q: cannot serialize object %T to JSON: %w", key, val, err))
		return
	}
	s.store(key, buf)
}

func unmarshal(val string, dst interface{}) error {
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

// Get a key,value pair from the setting.
func (s *Settings) Get(key string, dst interface{}) bool {
	buf := s.load(key)
	if buf == "" {
		return false
	}
	if err := unmarshal(buf, dst); err != nil {
		s.fErr(fmt.Errorf("cannot get setting %q=%s: %w", key, string(buf), err))
		return false
	}
	return true
}
