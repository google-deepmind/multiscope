// Copyright 2023 DeepMind Technologies Limited
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

package puller

import "multiscope/internal/settings"

type local struct {
	*settings.Base

	keyValues map[string]string
}

func newSettings(fErr func(error)) settings.Settings {
	l := &local{
		keyValues: make(map[string]string),
	}
	l.Base = settings.NewBase(l, fErr)
	return l
}

func (s *local) Delete(key string) error {
	delete(s.keyValues, key)
	return nil
}

func (s *local) Store(key string, buf []byte) error {
	s.keyValues[key] = string(buf)
	return nil
}

func (s *local) Load(key string) string {
	return s.keyValues[key]
}
