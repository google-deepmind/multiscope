package remote

import (
	"context"
	rootpb "multiscope/protos/root_go_proto"
	rootpbgrpc "multiscope/protos/root_go_proto"

	"google.golang.org/grpc"
)

// Display is the client to call functions on the server related to display on the dashboard.
type Display struct {
	clt rootpbgrpc.RootClient
	// Decides if new panels are added to the initial layout by default.
	globalDisplayByDefault bool

	list *rootpb.LayoutList
}

func newDisplay(conn grpc.ClientConnInterface) *Display {
	return &Display{
		clt:                    rootpbgrpc.NewRootClient(conn),
		globalDisplayByDefault: true,
		list:                   &rootpb.LayoutList{},
	}
}

// SetGlobalDisplayByDefault sets if new panels should be added to the initial dashboard by default.
func (d *Display) SetGlobalDisplayByDefault(onByDefault bool) {
	d.globalDisplayByDefault = onByDefault
}

// DisplayIfDefault will display the panel on the dashboard by default if enabled for this client.
func (d *Display) DisplayIfDefault(path Path) error {
	if !d.globalDisplayByDefault {
		return nil
	}
	d.list.Displayed = append(d.list.Displayed, path.NodePath())
	_, err := d.clt.SetLayout(context.Background(), &rootpb.SetLayoutRequest{
		Layout: &rootpb.Layout{
			Layout: &rootpb.Layout_List{
				List: d.list,
			},
		},
	})
	return err
}
