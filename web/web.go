// Package web returns a file system to access resources on the server.
package web

import (
	"embed"
	"io/fs"
	"os"
	"path/filepath"
)

//go:embed res/*
var content embed.FS

func findProjectRoot() string {
	cwd, err := os.Getwd()
	if err != nil {
		return ""
	}
	for cwd != "/" {
		modPath := filepath.Join(cwd, "go.mod")
		if _, err := os.Stat(modPath); err == nil {
			return cwd
		}
		cwd = filepath.Dir(cwd)
	}
	return ""
}

// FS returns the file system storing web content.
func FS() fs.FS {
	dir := findProjectRoot()
	if dir == "" {
		return content
	}
	webPath := filepath.Join(dir, "web")
	if _, err := os.Stat(webPath); err == nil {
		return os.DirFS(webPath)
	}
	return content
}
