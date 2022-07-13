package worker

import (
	"reflect"
	"runtime"
)

// MainFunc is a function invoked by a webworker.
type MainFunc func(wkr *Worker)

var registry = map[string]MainFunc{}

func (f MainFunc) name() string {
	return runtime.FuncForPC(reflect.ValueOf(f).Pointer()).Name()
}

// Register a function for a future invocation.
func Register(f MainFunc) {
	registry[f.name()] = f
}
