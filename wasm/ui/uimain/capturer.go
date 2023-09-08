package uimain

import (
	"syscall/js"

	eventspb "multiscope/protos/events_go_proto"

	"honnef.co/go/js/dom/v2"
)

// capturer captures keyboard events and send them to the client.
type capturer struct {
	ui *UI

	enabled         bool
	keyDownListener js.Func
	keyUpListener   js.Func
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
	c.keyDownListener = c.ui.window.AddEventListener("keydown", true, func(ev dom.Event) {
		c.sendKeyEvent(eventspb.Keyboard_DOWN, ev)
	})
	c.keyUpListener = c.ui.window.AddEventListener("keyup", true, func(ev dom.Event) {
		c.sendKeyEvent(eventspb.Keyboard_UP, ev)
	})
}

func (c *capturer) sendKeyEvent(typ eventspb.Keyboard_Type, ev dom.Event) {
	ev.StopPropagation()
	kev := ev.(*dom.KeyboardEvent)
	c.ui.SendToServer(nil, &eventspb.Keyboard{
		Key:   int32(kev.KeyCode()),
		Type:  typ,
		Alt:   kev.AltKey(),
		Ctrl:  kev.CtrlKey(),
		Shift: kev.ShiftKey(),
		Meta:  kev.MetaKey(),
	})
}

func (c *capturer) disable() {
	c.enabled = false
	c.ui.window.RemoveEventListener("keydown", true, c.keyDownListener)
	c.ui.window.RemoveEventListener("keyup", true, c.keyUpListener)
}
