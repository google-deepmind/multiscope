// Package settings stores user settings.
package settings

import (
	"encoding/json"
	"fmt"
	"strings"
	"syscall/js"
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

func (s *Settings) load(key string) []byte {
	val := s.storage.Get(prefix + key)
	if val.IsNull() || val.IsUndefined() {
		return nil
	}
	return []byte(val.String())
}

// Set a key,value pair in the settings.
func (s *Settings) Set(key string, val interface{}) {
	if val == nil {
		s.del(key)
		return
	}
	buf, err := json.Marshal(val)
	if err != nil {
		s.fErr(fmt.Errorf("cannot store key %q: cannot serialize object %T to JSON: %w", key, val, err))
		return
	}
	s.store(key, buf)
}

// Get a key,value pair from the setting.
func (s *Settings) Get(key string, dst interface{}) {
	buf := s.load(key)
	if buf == nil {
		return
	}
	if err := json.Unmarshal(buf, dst); err != nil {
		s.fErr(fmt.Errorf("cannot get setting %q=%s: %w", key, string(buf), err))
	}
}
