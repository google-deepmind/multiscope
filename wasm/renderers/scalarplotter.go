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

package renderers

import (
	"multiscope/internal/grpc/client"
	"multiscope/internal/style"
	plotpb "multiscope/protos/plot_go_proto"
	treepb "multiscope/protos/tree_go_proto"
	uipb "multiscope/protos/ui_go_proto"
	"syscall/js"
)

type scalarPlot struct {
	gplot *gonumPlot
}

func init() {
	Register(NewScalarPlot)
}

// NewScalarPlot returns a renderer to plot on a canvas using gonum/plot.
func NewScalarPlot(stl *style.Style, regPanel *uipb.RegisterPanel, aux js.Value) Renderer {
	return &scalarPlot{gplot: newGonumPlot(stl, regPanel, aux)}
}

func (rdr *scalarPlot) Render(data *treepb.NodeData) (*treepb.NodeData, error) {
	pbPlot := &plotpb.ScalarsPlot{}
	if err := client.ToProto(data, pbPlot); err != nil {
		return nil, err
	}
	if pbPlot.Plot == nil {
		return nil, nil
	}
	return rdr.gplot.renderPlot(pbPlot.Plot)
}
