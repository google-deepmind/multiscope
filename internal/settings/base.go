// Copyright 2023 Google LLC
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

package settings

import "fmt"

// Base settings to register listeners.
type Base struct {
	fErr      func(error)
	storage   Storage
	listeners map[string][]Listener
}

// NewBase returns a base settings.
func NewBase(storage Storage, fErr func(error)) *Base {
	return &Base{
		fErr:      fErr,
		storage:   storage,
		listeners: make(map[string][]Listener),
	}
}

func (s *Base) callListeners(buf string, src any, key string) error {
	listeners := s.listeners[key]
	for _, l := range listeners {
		if err := updateListener(src, key, buf, l); err != nil {
			return fmt.Errorf("settings listener error: %v", err)
		}
	}
	return nil
}

func (s *Base) set(src any, key string, val any) error {
	if val == nil {
		return s.storage.Delete(key)
	}
	buf, err := marshal(val)
	if err != nil {
		return fmt.Errorf("cannot store key %q: cannot serialize object %T to JSON: %w", key, val, err)
	}
	return s.storage.Store(key, buf)
}

// Set a key,value pair in the settings.
func (s *Base) Set(src any, key string, val any) {
	if err := s.set(src, key, val); err != nil {
		s.fErr(err)
	}
	s.update(src, key)
}

// Listen to a setting. The callback is called everytime the setting is changed.
func (s *Base) Listen(key string, dst any, f OnChangeListener) {
	listeners := s.listeners[key]
	l := Listener{Destination: dst, Callback: f}
	listeners = append(listeners, l)
	s.listeners[key] = listeners

	// Call the listener right away if the key is set.
	buf := s.storage.Load(key)
	if buf == "" {
		return
	}
	if err := updateListener(s, key, buf, l); err != nil {
		s.fErr(fmt.Errorf("settings listener error: %v", err))
	}
}

func (s *Base) update(src any, key string) {
	buf := s.storage.Load(key)
	if buf == "" {
		return
	}
	if err := s.callListeners(buf, src, key); err != nil {
		s.fErr(err)
	}
}

// CallAll calls all listeners for all keys.
func (s *Base) CallAll() {
	for key := range s.listeners {
		s.update(s, key)
	}
}
