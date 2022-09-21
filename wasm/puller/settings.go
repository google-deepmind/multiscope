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
