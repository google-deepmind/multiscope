// Template provides helper functions to execute Go templates.
package template

import (
	"fmt"
	"html/template"
	"io"
	"io/fs"
)

// Execute loads a file from a file system, parse it as a template,
// run the template with the data provided, and write the results
// to the provided writer.
func Execute(w io.Writer, root fs.FS, path string, data any) error {
	file, err := root.Open(path)
	if err != nil {
		return fmt.Errorf("error opening %q: %v", path, err)
	}
	defer file.Close()
	buf, err := io.ReadAll(file)
	if err != nil {
		return fmt.Errorf("cannot read %q: %v", path, err)
	}
	t, err := template.New(path).Parse(string(buf))
	if err != nil {
		return fmt.Errorf("error parsing template %q: %v", path, err)
	}
	if err = t.Execute(w, data); err != nil {
		return fmt.Errorf("error writing content of template %q: %v", path, err)
	}
	return nil
}
