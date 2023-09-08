// Copyright 2023 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package remote

import (
	"context"
	rootpb "multiscope/protos/root_go_proto"
	rootpbgrpc "multiscope/protos/root_go_proto"

	"github.com/pkg/errors"
)

// Display is the client to call functions on the server related to display on the dashboard.
type Display struct {
	clt        *Client
	rootClient rootpbgrpc.RootClient
	// Decides if new panels are added to the initial layout by default.
	globalDisplayByDefault bool

	list *rootpb.LayoutList
}

func newDisplay(clt *Client) *Display {
	return &Display{
		clt:                    clt,
		rootClient:             rootpbgrpc.NewRootClient(clt.conn),
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
	_, err := d.rootClient.SetLayout(context.Background(), &rootpb.SetLayoutRequest{
		TreeId: d.clt.treeID,
		Layout: &rootpb.Layout{
			Layout: &rootpb.Layout_List{
				List: d.list,
			},
		},
	})
	if err != nil {
		return errors.Errorf("cannot set display by default: %v", err)
	}
	return nil
}

// SetCapture enables or disables the capture button in the UI.
func (d *Display) SetCapture(enable bool) error {
	_, err := d.rootClient.SetCapture(
		context.Background(),
		&rootpb.SetCaptureRequest{
			TreeId:  d.clt.treeID,
			Capture: enable,
		},
	)
	return err
}
