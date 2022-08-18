// Package uimain implements the main page seen by the user, implementing
// the interfaces defined in the top ui package.
package uimain

import (
	"fmt"
	"multiscope/internal/css"
	"multiscope/internal/httpgrpc"
	"multiscope/internal/style"
	"multiscope/internal/wplot"
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
	"honnef.co/go/js/dom/v2"
)

// UI is the main page (i.e. user interface) with which the user interacts.
type UI struct {
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
func NewUI(puller *worker.Worker, c *uipb.Connect) *UI {
	ui := &UI{
		addr:   c,
		window: dom.GetWindow(),
	}
	ui.settings = settings.NewSettings(ui.DisplayErr)
	var err error
	ui.style, err = ui.newDefaultStyle()
	if err != nil {
		ui.DisplayErr(err)
		return ui
	}

	injector.Run(ui)
	ui.style.OnChange(func(s *style.Style) {
		ui.Owner().Body().Style().SetProperty("background", css.Color(s.Background()), "")
		ui.Owner().Body().Style().SetProperty("color", css.Color(s.Foreground()), "")
	})

	conn := httpgrpc.Connect(ui.addr.Scheme, ui.addr.Host)
	ui.treeClient = treepb.NewTreeClient(conn)
	ui.puller = newPuller(ui, puller)
	if ui.layout, err = newLayout(ui); err != nil {
		ui.DisplayErr(err)
		return ui
	}
	if err := ui.puller.registerPanel(ui.layout.Dashboard().descriptor()); err != nil {
		ui.DisplayErr(err)
		return ui
	}
	return ui
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

func (ui *UI) newDefaultStyle() (*style.Style, error) {
	s := style.NewStyle(ui.settings)
	body := ui.Owner().Body()
	css := dom.GetWindow().GetComputedStyle(body, "")
	fontSize, err := parseFontSize(css.GetPropertyValue("font-size"))
	if err != nil {
		return s, err
	}
	fontFamily := css.GetPropertyValue("font-family")
	s.Set("", fontFamily, fontSize)
	return s, nil
}

// Owner returns the owner of the DOM tree of the UI.
func (ui *UI) Owner() dom.HTMLDocument {
	return ui.window.Document().(dom.HTMLDocument)
}

// Dashboard returns the dashboard displaying the panels.
func (ui *UI) Dashboard() ui.Dashboard {
	return ui.layout.Dashboard()
}

// DisplayErr displays an error on the UI.
func (ui *UI) DisplayErr(err error) {
	if err.Error() == ui.lastError {
		return
	}
	ui.lastError = err.Error()
	fmt.Println("ERROR reported to main:")
	fmt.Println(err)
}

// Style returns the current style of the UI.
func (ui *UI) Style() *style.Style {
	return ui.style
}

// Layout returns the overall page layout.
func (ui *UI) Layout() *Layout {
	return ui.layout
}

// Settings returns Multiscope settings.
func (ui *UI) Settings() *settings.Settings {
	return ui.settings
}

func (ui *UI) renderFrame() error {
	displayData := ui.puller.lastDisplayData()
	if displayData == nil {
		return nil
	}
	if displayData.Err != "" {
		return fmt.Errorf("display data error: %v", displayData.Err)
	}
	ui.layout.Dashboard().render(displayData)
	return nil
}

func (ui *UI) animationFrame(period time.Duration) {
	if err := ui.renderFrame(); err != nil {
		ui.DisplayErr(err)
		return
	}
	ui.window.RequestAnimationFrame(ui.animationFrame)
}

// MainLoop runs the user interface main loop. It never returns.
func (ui *UI) MainLoop() {
	ui.window.RequestAnimationFrame(ui.animationFrame)
	<-make(chan bool)
}

// TreeClient returns the connection to the server.
func (ui *UI) TreeClient() treepb.TreeClient {
	return ui.treeClient
}

// Run a function in the background.
func (ui *UI) Run(f func() error) {
	go func() {
		if err := f(); err != nil {
			ui.DisplayErr(err)
		}
	}()
}
