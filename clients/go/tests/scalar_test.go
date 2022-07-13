package scope_test

import (
	"testing"

	"multiscope/clients/go/clienttesting"
	"multiscope/clients/go/remote"
	"multiscope/internal/server/writers/scalar/scalartesting"
)

func TestScalarWriter(t *testing.T) {
	clt, err := clienttesting.Start()
	if err != nil {
		t.Fatal(err)
	}
	scalarWriter, err := remote.NewScalarWriter(clt, scalartesting.Scalar01Name, nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, data := range scalartesting.Scalar01Data {
		if err := scalarWriter.WriteFloat64(data); err != nil {
			t.Fatal(err)
		}
	}
	if err := scalartesting.CheckScalar01(clt.TreeClient(), scalarWriter.Path()); err != nil {
		t.Error(err)
	}
}
