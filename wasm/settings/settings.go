// Package settings stores user settings.
package settings

import (
	"encoding/json"
	"fmt"
	"syscall/js"

	"multiscope/internal/settings"
)

const multiscope = "multiscope_wasm"

type (
	storage struct {
		settings map[string]map[string]string
		parent   *Settings
	}

	// Settings stores settings in cookies.
	Settings struct {
		*settings.Base
		settings storage
		dictKey  string
		storage  js.Value
	}
)

var _ settings.Settings = (*Settings)(nil)

// NewSettings returns a new instance of Settings.
func NewSettings(dictKey string, fErr func(error)) *Settings {
	s := &Settings{
		storage: js.Global().Get("localStorage"),
		dictKey: dictKey,
	}
	s.settings = storage{
		settings: make(map[string]map[string]string),
		parent:   s,
	}
	s.Base = settings.NewBase(&s.settings, fErr)

	val := s.storage.Get(multiscope)
	if val.IsNull() || val.IsUndefined() {
		return s
	}
	if err := json.Unmarshal([]byte(val.String()), &s.settings.settings); err != nil {
		fErr(fmt.Errorf("cannot load settings from frontend storage: %v", err))
	}
	return s
}

// SetDictKey sets the key used to fetch the dictionary of settings.
func (s *Settings) SetDictKey(dict string) {
	if dict == s.dictKey {
		return
	}
	s.dictKey = dict
	s.Base.CallAll()
}

func (s *Settings) updateStorage() error {
	buf, err := json.Marshal(s.settings.settings)
	if err != nil {
		return fmt.Errorf("cannot update frontend storage: %v", err)
	}
	s.storage.Set(multiscope, string(buf))
	return nil
}

func (s *storage) Delete(key string) error {
	keyValues := s.settings[s.parent.dictKey]
	if keyValues == nil {
		return nil
	}
	delete(keyValues, key)
	return s.parent.updateStorage()
}

func (s *storage) Store(key string, buf []byte) error {
	keyValues := s.settings[s.parent.dictKey]
	if keyValues == nil {
		keyValues = make(map[string]string)
		s.settings[s.parent.dictKey] = keyValues
	}
	keyValues[key] = string(buf)
	return s.parent.updateStorage()
}

func (s *storage) Load(key string) string {
	keyValues := s.settings[s.parent.dictKey]
	if keyValues == nil {
		return ""
	}
	return keyValues[key]
}
