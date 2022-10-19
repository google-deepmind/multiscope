package fmtx

import (
	"fmt"

	"github.com/pkg/errors"
)

type stackTracer interface {
	StackTrace() errors.StackTrace
}

const coreError = "\n\n\n===========\nCORE ERROR:\n"

func formatCoreError(err error) string {
	last := err
	for unwrap := last; unwrap != nil; unwrap = errors.Unwrap(unwrap) {
		last = unwrap
	}
	if last == err {
		return ""
	}
	return coreError + last.Error()
}

// FormatError formats an error with its stacktrace if available.
func FormatError(err error) error {
	if err == nil {
		return nil
	}
	s := err.Error()
	var withSt stackTracer
	if errors.As(err, &withSt) {
		s += fmt.Sprintf("%s%+v\n", coreError, withSt.StackTrace())
	} else {
		s += formatCoreError(err)
	}
	return errors.New(s)
}
