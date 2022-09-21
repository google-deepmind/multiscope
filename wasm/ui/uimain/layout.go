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
