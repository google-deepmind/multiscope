// Copyright 2023 Google LLC
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

package tensor

import (
	"multiscope/internal/server/core"
	"multiscope/internal/server/writers/scalar"
)

type metricsUpdater struct {
	indexer
	parent *Writer
	minmax *scalar.Writer
	norms  *scalar.Writer
}

func newMetrics(parent *Writer) *metricsUpdater {
	minmax := scalar.NewWriter()
	parent.AddChild(NodeNameMinMax, minmax)
	norms := scalar.NewWriter()
	parent.AddChild(NodeNameNorms, norms)
	return &metricsUpdater{
		parent: parent,
		minmax: minmax,
		norms:  norms,
	}
}

func (u *metricsUpdater) reset() error {
	u.minmax.Reset()
	u.norms.Reset()
	return nil
}

func (u *metricsUpdater) forwardActive(parent *core.Path) {
	// Forward the activity of the minmax panel.
	minmaxPath := parent.PathTo(NodeNameMinMax)
	u.parent.state.PathLog().Forward(parent, minmaxPath)

	// Forward the activity of the norm panel.
	normsPath := parent.PathTo(NodeNameNorms)
	u.parent.state.PathLog().Forward(parent, normsPath)
}

func (u *metricsUpdater) update(updateIndex uint, t sTensor) (err error) {
	return u.forceUpdate(updateIndex, t)
}

func (u *metricsUpdater) forceUpdate(updateIndex uint, _ sTensor) (err error) {
	u.updateIndex(updateIndex)
	if err := u.minmax.Write(map[string]float64{
		"min": float64(u.parent.m.Min),
		"max": float64(u.parent.m.Max),
	}); err != nil {
		return err
	}
	return u.norms.Write(map[string]float64{
		"l1norm": float64(u.parent.m.L1Norm),
		"l2norm": float64(u.parent.m.L2Norm),
	})
}
