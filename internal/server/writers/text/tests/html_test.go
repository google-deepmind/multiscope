package text_test

import (
	"testing"

	"multiscope/clients/go/clienttesting"
	"multiscope/clients/go/remote"
	"multiscope/internal/server/writers/text/texttesting"
)

func TestHTMLWriter(t *testing.T) {
	clt, err := clienttesting.Start()
	if err != nil {
		t.Fatal(err)
	}
	writer, err := remote.NewHTMLWriter(clt, texttesting.HTML01Name, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := writer.WriteCSS(texttesting.CSS01Data); err != nil {
		t.Fatal(err)
	}
	for i, want := range texttesting.HTML01Data {
		if err := writer.Write(want); err != nil {
			t.Error(err)
			break
		}
		if err := texttesting.CheckHTML01(clt.TreeClient(), []string{texttesting.Text01Name}, i); err != nil {
			t.Error(err)
		}
	}
}
