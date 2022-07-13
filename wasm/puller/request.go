package puller

import (
	"multiscope/internal/server/core"
	treepb "multiscope/protos/tree_go_proto"
	uipb "multiscope/protos/ui_go_proto"
	"multiscope/wasm/renderers"
	"multiscope/wasm/ui"
)

type (
	panelS struct {
		pb  *uipb.Panel
		rdr renderers.Renderer
	}

	pathPointers struct {
		req    *treepb.DataRequest
		panels map[ui.PanelID]*panelS
	}

	request struct {
		paths   map[core.Key]*pathPointers
		request *treepb.NodeDataRequest
	}
)

func newRequest() *request {
	return &request{
		paths:   make(map[core.Key]*pathPointers),
		request: &treepb.NodeDataRequest{},
	}
}

func (r *request) registerPath(panel *panelS, path *treepb.NodePath) {
	key := core.ToKey(path.Path)
	pointers := r.paths[key]
	if pointers == nil {
		pointers = &pathPointers{}
		pointers.req = &treepb.DataRequest{
			Path: path,
		}
		pointers.panels = make(map[ui.PanelID]*panelS)
		r.paths[key] = pointers
		r.request.Reqs = append(r.request.Reqs, pointers.req)
	}
	pointers.panels[ui.PanelID(panel.pb.Id)] = panel
}

func (r *request) setLastTick(path *treepb.NodePath, lastTick uint32) {
	if path == nil {
		return
	}
	key := core.ToKey(path.Path)
	pointers := r.paths[key]
	if pointers == nil {
		return
	}
	pointers.req.LastTick = lastTick
}

func (r *request) panels(path *treepb.NodePath) []*panelS {
	if path == nil {
		return nil
	}
	pointers := r.paths[core.ToKey(path.Path)]
	res := make([]*panelS, 0, len(pointers.panels))
	for _, panel := range pointers.panels {
		res = append(res, panel)
	}
	return res
}

func (r *request) pb() *treepb.NodeDataRequest {
	return r.request
}
