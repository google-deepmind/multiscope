package wplot

import "gonum.org/v1/plot"

type (
	// AxisOption is an option to apply on a axis.
	AxisOption interface {
		Apply(*plot.Axis)
	}

	// AxisOptionFunc applies settings to an axis.
	AxisOptionFunc func(*plot.Axis)
)

// Apply the settings to the axis.
func (f AxisOptionFunc) Apply(ax *plot.Axis) {
	f(ax)
}

// Min sets the minimum value of an axis.
func Min(m float64) AxisOption {
	return AxisOptionFunc(func(ax *plot.Axis) {
		ax.Min = m
	})
}

// Max sets the maximum value of an axis.
func Max(m float64) AxisOption {
	return AxisOptionFunc(func(ax *plot.Axis) {
		ax.Max = m
	})
}

// Label sets the label of an axis.
func Label(l string) AxisOption {
	return AxisOptionFunc(func(ax *plot.Axis) {
		ax.Label.Text = l
	})
}

type logScale struct{}

// HasLogScale returns true if one of the option is a log scale option.
func HasLogScale(opts []AxisOption) bool {
	for _, o := range opts {
		if _, ok := o.(logScale); ok {
			return true
		}
	}
	return false
}

// Apply the settings to the axis.
func (logScale) Apply(ax *plot.Axis) {
	ax.Scale = plot.LogScale{}
	ax.Tick.Marker = plot.LogTicks{}
}

// LogScale sets a log scale for the axis.
func LogScale() AxisOption {
	return logScale{}
}

func applyAxisOptions(ax *plot.Axis, opts []AxisOption) {
	for _, o := range opts {
		o.Apply(ax)
	}
}
