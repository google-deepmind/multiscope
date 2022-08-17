package puller

import (
	"fmt"
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

func findReqPos(reqs []*treepb.DataRequest, req *treepb.DataRequest) int {
	for i, reqi := range reqs {
		if reqi == req {
			return i
		}
	}
	return -1
}

func (r *request) unregisterPath(id ui.PanelID, path *treepb.NodePath) error {
	key := core.ToKey(path.Path)
	pointers := r.paths[key]
	if pointers == nil {
		return nil
	}
	delete(pointers.panels, id)
	if len(pointers.panels) > 0 {
		return nil
	}
	delete(r.paths, key)
	i := findReqPos(r.request.Reqs, pointers.req)
	if i < 0 {
		return fmt.Errorf("unexpected error: could not unregister %q", key)
	}
	r.request.Reqs[i] = r.request.Reqs[len(r.request.Reqs)-1]
	r.request.Reqs = r.request.Reqs[:len(r.request.Reqs)-1]
	return nil
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
