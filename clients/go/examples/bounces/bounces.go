// bounce is an example demonstrating how to generate images and plots.
package main

import (
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"log"
	"multiscope/clients/go/scope"
	"runtime"
	"sync"
)

var (
	httpPort = flag.Int("http_port", scope.DefaultPort, "http port")
	local    = flag.Bool("local", true, "open the port to local connection only")

	numBounces = flag.Int("nums", 3, "number of parallel bouncing balls")
)

type box struct {
	pos  image.Point
	size int
}

func (b *box) ColorModel() color.Model {
	return color.AlphaModel
}

func (b *box) Bounds() image.Rectangle {
	return image.Rect(b.pos.X, b.pos.Y, b.pos.X+b.size, b.pos.Y+b.size)
}

func (b *box) At(x, y int) color.Color {
	if x > b.pos.X && x < b.pos.X+b.size && y > b.pos.Y && y < b.pos.Y+b.size {
		return color.Alpha{255}
	}
	return color.Alpha{0}
}

type bounce struct {
	ticker         *scope.Ticker
	bl             *box
	deltaX, deltaY int
	sizeX, sizeY   int
	img            *image.RGBA
}

func newBounce(ticker *scope.Ticker, size, deltaX, deltaY, sizeX, sizeY int) *bounce {
	b := &box{image.Point{sizeX / 2, sizeY / 2}, size}
	return &bounce{
		ticker: ticker,
		bl:     b,
		deltaX: deltaX,
		deltaY: deltaY,
		sizeX:  sizeX,
		sizeY:  sizeY,
	}
}

func (b *bounce) Step() {
	x := b.bl.pos.X + b.deltaX
	y := b.bl.pos.Y + b.deltaY

	if x < 0 {
		x = 0
		b.deltaX *= -1
	}
	if y < 0 {
		y = 0
		b.deltaY *= -1
	}

	if x+b.bl.size >= b.sizeX {
		x = b.sizeX - b.bl.size
		b.deltaX *= -1
	}
	if y+b.bl.size >= b.sizeY {
		y = b.sizeY - b.bl.size
		b.deltaY *= -1
	}

	b.bl.pos.X = x
	b.bl.pos.Y = y
	b.ticker.Tick()
}

func (b *bounce) UpdateImage() *image.RGBA {
	if b.img == nil {
		b.img = image.NewRGBA(image.Rect(0, 0, b.sizeX, b.sizeY))
	}
	background := color.RGBA{0, 0, 255, 255}
	draw.Draw(b.img, b.img.Bounds(), &image.Uniform{background}, image.ZP, draw.Src)
	box := color.RGBA{255, 0, 0, 255}
	draw.DrawMask(b.img, b.img.Bounds(), &image.Uniform{box}, image.ZP, b.bl, image.ZP, draw.Over)
	return b.img
}

func runBounce(wg *sync.WaitGroup, name string, size int) {
	ticker, err := scope.NewTicker(name, nil)
	if err != nil {
		log.Fatal(err)
	}
	b := newBounce(ticker, size, 3, 4, 200, 300)
	w1, err := scope.NewImageWriter("image", ticker.Path())
	if err != nil {
		log.Fatal(err)
	}
	w2, err := scope.NewScalarWriter("values", ticker.Path())
	if err != nil {
		log.Fatal(err)
	}
	w3, err := scope.NewTextWriter("text", ticker.Path())
	if err != nil {
		log.Fatal(err)
	}
	wg.Done()
	for true {
		w1.Write(b.UpdateImage())
		w2.Write(map[string]interface{}{
			"x": b.bl.pos.X,
			"y": b.bl.pos.Y,
		})
		w3.Write(fmt.Sprintf("x: %d y: %d", b.bl.pos.X, b.bl.pos.Y))
		b.Step()
	}
}

func main() {
	flag.Parse()
	runtime.GOMAXPROCS(2)
	if err := scope.Start(*httpPort, *local); err != nil {
		log.Fatal(err)
	}

	wg := sync.WaitGroup{}
	for i := 0; i < *numBounces; i++ {
		wg.Wait()
		wg.Add(1)
		name := fmt.Sprintf("Bounce %d", i)
		size := (i + 1) * 10
		go runBounce(&wg, name, size)
	}
	<-make(chan bool)
}
