// Package uimain implements the main page seen by the user, implementing
// the interfaces defined in the top ui package.
package uimain

import (
	"context"
	"fmt"
	"multiscope/internal/css"
	"multiscope/internal/fmtx"
	"multiscope/internal/httpgrpc"
	"multiscope/internal/style"
	"multiscope/internal/wplot"
	rootpb "multiscope/protos/root_go_proto"
	treepb "multiscope/protos/tree_go_proto"
	uipb "multiscope/protos/ui_go_proto"
	"multiscope/wasm/injector"
	"multiscope/wasm/settings"
	"multiscope/wasm/ui"
	"multiscope/wasm/worker"
	"strconv"
	"strings"
	"time"

	"github.com/pkg/errors"
	"gonum.org/v1/plot/vg"
	"google.golang.org/grpc"
	"honnef.co/go/js/dom/v2"
)

// UI is the main page (i.e. user interface) with which the user interacts.
type UI struct {
	owner      *ui.Owner
	window     dom.Window
	addr       *uipb.Connect
	treeClient treepb.TreeClient
	style      *style.Style
	settings   *settings.Settings
	puller     *puller
	layout     *Layout

	lastError string
}

// NewUI returns a new user interface mananing the main page.
func NewUI(pullerWorker *worker.Worker, c *uipb.Connect) *UI {
	gui := &UI{
		addr:   c,
		window: dom.GetWindow(),
	}
	gui.owner = ui.NewOwner(gui.window.Document().(dom.HTMLDocument))
	conn := httpgrpc.Connect(gui.addr.Scheme, gui.addr.Host)
	rootInfo, err := fetchRootInfo(conn)
	if err != nil {
		gui.DisplayErr(err)
	}
	gui.settings = settings.NewSettings(rootInfo.KeySettings, gui.DisplayErr)
	if gui.style, err = gui.newDefaultStyle(); err != nil {
		gui.DisplayErr(err)
		return gui
	}

	injector.Run(gui)
	gui.style.OnChange(func(s *style.Style) {
		setCSS(gui.Owner().Doc(), s)
	})

	gui.treeClient = treepb.NewTreeClient(conn)
	gui.puller = newPuller(gui, pullerWorker)
	if gui.layout, err = newLayout(gui, rootInfo); err != nil {
		gui.DisplayErr(err)
		return gui
	}
	dbd := gui.layout.Dashboard()
	if err := gui.puller.registerPanel(dbd.descriptor(), dbd.layout); err != nil {
		gui.DisplayErr(err)
		return gui
	}
	gui.style.OnChange(gui.onStyleChange)
	return gui
}

func setCSS(doc dom.HTMLDocument, s *style.Style) {
	root := doc.QuerySelector(":root").(dom.HTMLElement)
	root.Style().SetProperty("--main-fg-color", css.Color(s.Foreground()), "")
	root.Style().SetProperty("--main-bg-color", css.Color(s.Background()), "")
	root.Style().SetProperty("--main-bg-sub-color", css.Color(s.BackgroundSub()), "")
}

// SendToRenderers sends an event to renderers.
func (gui *UI) SendToRenderers(ev *uipb.UIEvent) {
	if err := gui.puller.toRenderers(ev); err != nil {
		gui.DisplayErr(err)
	}
}

func (gui *UI) onStyleChange(s *style.Style) {
	gui.SendToRenderers(&uipb.UIEvent{
		Event: &uipb.UIEvent_Style{
			Style: &uipb.StyleChange{
				Theme:      s.Theme().Name,
				FontSize:   float64(s.FontSize()),
				FontFamily: s.FontFamily(),
			},
		},
	})
}

func fetchRootInfo(conn grpc.ClientConnInterface) (*rootpb.RootInfo, error) {
	rootClient := rootpb.NewRootClient(conn)
	resp, err := rootClient.GetRootInfo(context.Background(), &rootpb.GetRootInfoRequest{})
	if err != nil {
		return &rootpb.RootInfo{}, err
	}
	return resp.Info, nil
}

func parseFontSize(s string) (size vg.Length, err error) {
	defer func() {
		if err != nil {
			err = fmt.Errorf("cannot parse font size property %q: %s", s, err)
		}
	}()
	if !strings.HasSuffix(s, "px") {
		return -1, errors.Errorf("size does not have pixel (px) units")
	}
	num := s[:len(s)-len("px")]
	f, err := strconv.ParseFloat(num, 64)
	if err != nil {
		return -1, errors.Errorf("cannot parse float %q: %v", num, err)
	}
	return wplot.ToLengthF(f), nil
}

func (gui *UI) newDefaultStyle() (*style.Style, error) {
	s := style.NewStyle(gui.settings)
	body := gui.Owner().Doc().Body()
	bodyCSS := dom.GetWindow().GetComputedStyle(body, "")
	fontSize, err := parseFontSize(bodyCSS.GetPropertyValue("font-size"))
	if err != nil {
		return s, err
	}
	fontFamily := bodyCSS.GetPropertyValue("font-family")
	s.Set("", fontFamily, fontSize)
	return s, nil
}

// Owner returns the owner of the DOM tree of the UI.
func (gui *UI) Owner() *ui.Owner {
	return gui.owner
}

// Dashboard returns the dashboard displaying the panels.
func (gui *UI) Dashboard() ui.Dashboard {
	return gui.layout.Dashboard()
}

// DisplayErr displays an error on the UI.
func (gui *UI) DisplayErr(err error) {
	err = fmtx.FormatError(err)
	if err.Error() == gui.lastError {
		return
	}
	gui.lastError = err.Error()
	fmt.Println("ERROR reported to main:")
	fmt.Println(err)
}

// Style returns the current style of the UI.
func (gui *UI) Style() *style.Style {
	return gui.style
}

// Layout returns the overall page layout.
func (gui *UI) Layout() *Layout {
	return gui.layout
}

// Settings returns Multiscope settings.
func (gui *UI) Settings() *settings.Settings {
	return gui.settings
}

func (gui *UI) renderFrame() error {
	displayData := gui.puller.lastDisplayData()
	if displayData == nil {
		return nil
	}
	if displayData.Err != "" {
		return fmt.Errorf("display data error: %s", displayData.Err)
	}
	gui.layout.Dashboard().render(displayData)
	return nil
}

func (gui *UI) animationFrame(period time.Duration) {
	if err := gui.renderFrame(); err != nil {
		gui.DisplayErr(err)
		return
	}
	gui.window.RequestAnimationFrame(gui.animationFrame)
}

// MainLoop runs the user interface main loop. It never returns.
func (gui *UI) MainLoop() {
	gui.window.RequestAnimationFrame(gui.animationFrame)
	<-make(chan bool)
}

// TreeClient returns the connection to the server.
func (gui *UI) TreeClient() treepb.TreeClient {
	return gui.treeClient
}

// Run a function in the background.
func (gui *UI) Run(f func() error) {
	go func() {
		if err := f(); err != nil {
			gui.DisplayErr(err)
		}
	}()
}
