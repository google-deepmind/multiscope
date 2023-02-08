package tensor

import (
	"fmt"
	"log"
)

func computeIndex(offsets []int, p []int) int {
	if len(offsets) != len(p) {
		log.Fatal("len(offsets) != len(p)")
	}
	index := 0
	for i, v := range p {
		index += offsets[i] * v
	}
	return index
}

func buildMatrixLabel(p []int) string {
	result := "["
	for _, v := range p {
		result += fmt.Sprint(v) + ","
	}
	return result + ".,.]=\n"
}

func buildMatrixString(t sTensor, offsets []int, p []int) string {
	shape := t.Shape()
	fullPos := make([]int, len(shape))
	copy(fullPos, p)
	result := ""
	if len(p) > 0 {
		result += buildMatrixLabel(p)
	}
	data := t.ValuesF32()
	for x := 0; x < shape[len(shape)-2]; x++ {
		for y := 0; y < shape[len(shape)-1]; y++ {
			fullPos[len(fullPos)-2] = x
			fullPos[len(fullPos)-1] = y
			result += fmt.Sprintf("%v,", data[computeIndex(offsets, fullPos)])
		}
		result += "\n"
	}
	return result
}

func buildStringRec(t sTensor, offsets []int, p []int) string {
	shape := t.Shape()
	if len(shape)-len(p) == 2 {
		return buildMatrixString(t, offsets, p)
	}
	currentDimSize := shape[len(p)]
	currentPosition := append(append([]int{}, p...), 0)
	result := ""
	for i := 0; i < currentDimSize; i++ {
		currentPosition[len(currentPosition)-1] = i
		result += buildStringRec(t, offsets, currentPosition)
	}
	return result
}

func computeOffsets(shape []int) []int {
	offsets := make([]int, len(shape))
	for i := range offsets {
		offsets[i] = 1
		for _, d := range shape[i+1:] {
			offsets[i] *= d
		}
	}
	return offsets
}

func toString(t sTensor) string {
	if len(t.Shape()) <= 1 {
		return fmt.Sprint(t.ValuesF32())
	}
	return buildStringRec(t, computeOffsets(t.Shape()), []int{})
}
