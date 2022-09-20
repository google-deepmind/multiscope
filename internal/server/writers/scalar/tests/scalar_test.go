package scalar_test

import (
	"testing"

	"multiscope/internal/grpc/grpctesting"
	"multiscope/internal/server/root"
	"multiscope/internal/server/writers/scalar"
	"multiscope/internal/server/writers/scalar/scalartesting"
)

func TestScalarWriter(t *testing.T) {
	rootNode := root.NewRoot()
	state := grpctesting.NewState(rootNode, nil, nil)
	conn, clt, err := grpctesting.SetupTest(state)
	if err != nil {
		t.Fatal(err)
		return
	}
	defer conn.Close()
	scalarWriter := scalar.NewWriter()
	rootNode.AddChild(scalartesting.Scalar01Name, scalarWriter)
	for _, v := range scalartesting.Scalar01Data {
		if err := scalarWriter.Write(v); err != nil {
			t.Fatal(err)
		}
	}
	if err := scalartesting.CheckScalar01(clt, []string{scalartesting.Scalar01Name}); err != nil {
		t.Error(err)
	}
}
