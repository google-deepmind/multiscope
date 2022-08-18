// Package settings stores user settings.
package settings

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"reflect"
	"syscall/js"

	"honnef.co/go/js/dom/v2"
)

// Settings stores settings in cookies.
type Settings struct {
	fErr func(error)
	cur  map[string]interface{}
}

func NewSettings(fErr func(error)) *Settings {
	s := &Settings{
		fErr: fErr,
		cur:  make(map[string]interface{}),
	}

	js.Global().Set("multiscopeSettings", js.FuncOf(func(js.Value, []js.Value) interface{} {
		go s.printSettings()
		return nil
	}))

	header := http.Header{}
	header.Add("Cookie", doc().Cookie())
	req := http.Request{Header: header}
	cookie, err := req.Cookie("settings")
	if err != nil || cookie == nil {
		return s
	}
	decoded, err := base64.StdEncoding.WithPadding(base64.NoPadding).DecodeString(cookie.Value)
	if err != nil {
		s.fErr(fmt.Errorf("cannot decode base64 settings: %v", err))
		return s
	}
	if err := json.Unmarshal(decoded, &s.cur); err != nil {
		s.fErr(fmt.Errorf("cannot decode JSON settings: %v", err))
		return s
	}
	return s
}

func (s *Settings) printSettings() {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetIndent("", "  ")
	if err := enc.Encode(s.cur); err != nil {
		fmt.Println("cannot serialize the settings:", err)
		return
	}
	fmt.Println(buf.String())
}

func doc() dom.HTMLDocument {
	return dom.GetWindow().Document().(dom.HTMLDocument)
}

// Set a key,value pair in the settings.
func (s *Settings) Set(key string, val interface{}) {
	if val == nil {
		delete(s.cur, key)
	} else {
		s.cur[key] = val
	}
	stg, err := json.Marshal(s.cur)
	if err != nil {
		delete(s.cur, key)
		s.fErr(fmt.Errorf("cannot serialize the settings to JSON after setting key %q=%v: %w", key, val, err))
		return
	}
	encoded := base64.StdEncoding.WithPadding(base64.NoPadding).EncodeToString([]byte(stg))
	doc().SetCookie(fmt.Sprintf("%s=%s; Tue, 19 Jan 2038 04:14:07 GMT", "settings", encoded))
}

func dynamicCast(v reflect.Value, dstType reflect.Type) reflect.Value {
	switch dstType.Kind() {
	case reflect.Bool:
		r, ok := v.Interface().(bool)
		if !ok {
			return v
		}
		return reflect.ValueOf(r)
	case reflect.String:
		r, ok := v.Interface().(string)
		if !ok {
			return v
		}
		return reflect.ValueOf(r)
	case reflect.Int:
		r, ok := v.Interface().(int)
		if !ok {
			return v
		}
		return reflect.ValueOf(r)
	case reflect.Float32:
		r, ok := v.Interface().(float32)
		if !ok {
			return v
		}
		return reflect.ValueOf(r)
	}
	return v
}

func copyMapToMap(src, dst reflect.Value) (err error) {
	defer func() {
		if err != nil {
			err = fmt.Errorf("cannot copy map from %s to %s: %w", src.Type().String(), dst.Type().String(), err)
		}
	}()
	dstKeyType := dst.Type().Key()
	dstValType := dst.Type().Elem()
	rang := src.MapRange()
	for rang.Next() {
		srcKeyType := rang.Key().Type()
		if !srcKeyType.ConvertibleTo(dstKeyType) {
			return fmt.Errorf("cannot convert key %s to key %s", srcKeyType.String(), dstKeyType.String())
		}
		value := rang.Value()
		if !value.Type().ConvertibleTo(dstValType) {
			value = dynamicCast(value, dstValType)
		}
		if !value.Type().ConvertibleTo(dstValType) {
			return fmt.Errorf("cannot convert value %s to value %s", value.Type().String(), dstValType.String())
		}
		dst.SetMapIndex(rang.Key().Convert(dstKeyType), value.Convert(dstValType))
	}
	return nil
}

func (s *Settings) get(key string, dst interface{}) error {
	src := s.cur[key]
	if src == nil {
		return nil
	}
	srcType := reflect.TypeOf(src)
	if srcType.Kind() == reflect.Map && reflect.TypeOf(dst).Kind() == reflect.Map {
		return copyMapToMap(reflect.ValueOf(src), reflect.ValueOf(dst))
	}
	dstVal := reflect.ValueOf(dst).Elem()
	dstType := dstVal.Type()
	if srcType.Kind() == reflect.Map && dstType.Kind() == reflect.Map {
		return copyMapToMap(reflect.ValueOf(src), dstVal)
	}
	if !srcType.ConvertibleTo(dstType) {
		return fmt.Errorf("cannot convert type %s to type %s", srcType.String(), dstType.String())
	}
	dstVal.Set(reflect.ValueOf(src).Convert(dstType))
	return nil
}

// Get a key,value pair from the setting.
func (s *Settings) Get(key string, dst interface{}) {
	if err := s.get(key, dst); err != nil {
		s.fErr(fmt.Errorf("cannot get setting %q: %w", key, err))
	}
}
