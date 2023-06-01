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

package uimain

import rootpb "multiscope/protos/root_go_proto"

// Layout owns the different components of a page.
type Layout struct {
	header *Header
	left   *LeftBar
	dbd    *Dashboard
}

func newLayout(ui *UI, rootInfo *rootpb.RootInfo) (*Layout, error) {
	l := &Layout{}
	var err error
	if l.header, err = newHeader(ui); err != nil {
		return nil, err
	}
	if l.dbd, err = newDashboard(ui, rootInfo); err != nil {
		return nil, err
	}
	if l.left, err = newLeftBar(ui); err != nil {
		return nil, err
	}
	return l, nil
}

// LeftBar returns the left bar of the page.
func (l *Layout) LeftBar() *LeftBar {
	return l.left
}

// Dashboard returns the element containing all the panels.
func (l *Layout) Dashboard() *Dashboard {
	return l.dbd
}
