package uimain

import (
	"syscall/js"

	eventspb "multiscope/protos/events_go_proto"

	"honnef.co/go/js/dom/v2"
)

// capturer captures keyboard events and send them to the client.
type capturer struct {
	ui *UI

	enabled bool
	// Keyboard
	keyDownListener js.Func
	keyUpListener   js.Func
	// Mouse
	mouseMoveListener js.Func
	mouseDownListener js.Func
	mouseUpListener   js.Func
}

func (c *capturer) Toggle() bool {
	if c.enabled {
		c.disable()
	} else {
		c.enable()
	}
	return c.enabled
}

func (c *capturer) enable() {
	c.enabled = true
	// Keyboard listeners.
	c.keyDownListener = c.ui.window.AddEventListener("keydown", true, func(ev dom.Event) {
		c.sendKeyEvent(eventspb.Keyboard_DOWN, ev)
	})
	c.keyUpListener = c.ui.window.AddEventListener("keyup", true, func(ev dom.Event) {
		c.sendKeyEvent(eventspb.Keyboard_UP, ev)
	})
	// Mouse listeners.
	c.mouseMoveListener = c.ui.window.AddEventListener("mousemove", true, func(ev dom.Event) {
		c.sendMouseEvent(eventspb.Mouse_MOVE, ev)
	})
	c.mouseDownListener = c.ui.window.AddEventListener("mousedown", true, func(ev dom.Event) {
		c.sendMouseEvent(eventspb.Mouse_DOWN, ev)
	})
	c.mouseUpListener = c.ui.window.AddEventListener("mouseup", true, func(ev dom.Event) {
		c.sendMouseEvent(eventspb.Mouse_UP, ev)
	})
}

func (c *capturer) sendKeyEvent(typ eventspb.Keyboard_Type, ev dom.Event) {
	ev.StopPropagation()
	kev := ev.(*dom.KeyboardEvent)
	c.ui.SendToServer(nil, &eventspb.Keyboard{
		Type:  typ,
		Key:   int32(kev.KeyCode()),
		Alt:   kev.AltKey(),
		Ctrl:  kev.CtrlKey(),
		Shift: kev.ShiftKey(),
		Meta:  kev.MetaKey(),
	})
}

func (c *capturer) sendMouseEvent(typ eventspb.Mouse_Type, ev dom.Event) {
	ev.StopPropagation()
	mev := ev.(*dom.MouseEvent)
	c.ui.SendToServer(nil, &eventspb.Mouse{
		Type:         typ,
		Key:          int32(mev.Button()),
		PositionX:    int32(mev.ClientX()),
		PositionY:    int32(mev.ClientY()),
		TranslationX: int32(mev.MovementX()),
		TranslationY: int32(mev.MovementY()),
	})
}

func (c *capturer) disable() {
	c.enabled = false
	c.ui.window.RemoveEventListener("keydown", true, c.keyDownListener)
	c.ui.window.RemoveEventListener("keyup", true, c.keyUpListener)
	c.ui.window.RemoveEventListener("mousemove", true, c.mouseMoveListener)
	c.ui.window.RemoveEventListener("mousedown", true, c.mouseDownListener)
	c.ui.window.RemoveEventListener("mouseup", true, c.mouseUpListener)
}
