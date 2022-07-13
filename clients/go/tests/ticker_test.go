package scope_test

import (
	"multiscope/clients/go/clienttesting"
	"multiscope/clients/go/remote"
	"multiscope/internal/server/writers/ticker/tickertesting"
	"testing"
)

func TestTicker(t *testing.T) {
	clt, err := clienttesting.Start()
	if err != nil {
		t.Fatal(err)
	}
	ticker, err := remote.NewTicker(clt, tickertesting.Ticker01Name, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := clienttesting.ForceActive(clt.TreeClient(), ticker.Path()); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 42; i++ {
		if err := ticker.Tick(); err != nil {
			t.Fatal(err)
		}
	}
	if err := tickertesting.CheckRemoteTicker(clt.TreeClient(), ticker.Path().NodePath().GetPath(), ticker.CurrentTick()); err != nil {
		t.Errorf("wrong data on the server: %v", err)
	}
}
