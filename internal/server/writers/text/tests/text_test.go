package text_test

import (
	"testing"

	"multiscope/clients/go/clienttesting"
	"multiscope/clients/go/remote"
	"multiscope/internal/server/writers/text/texttesting"
)

func TestWriter(t *testing.T) {
	clt, err := clienttesting.Start()
	if err != nil {
		t.Fatal(err)
	}
	writer, err := remote.NewTextWriter(clt, texttesting.Text01Name, nil)
	if err != nil {
		t.Fatal(err)
	}
	for i, want := range texttesting.Text01Data {
		if err := writer.Write(want); err != nil {
			t.Error(err)
			break
		}
		if err := texttesting.CheckText01(clt.TreeClient(), []string{texttesting.Text01Name}, i); err != nil {
			t.Error(err)
		}
	}
}
