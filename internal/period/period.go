// Package period provides helper functions to measure time period.
package period

import "time"

// Period measures the time period between samples.
type Period struct {
	HistoryTrace  float64
	lastSample    time.Time
	average, last time.Duration
}

// Sample registers a measurement to measure the period of.
func (p *Period) Sample() time.Duration {
	if p.lastSample.IsZero() {
		p.Start()
		return 0
	}
	now := time.Now()
	p.last = now.Sub(p.lastSample)
	p.average = time.Duration(p.HistoryTrace*float64(p.average) + (1-p.HistoryTrace)*float64(p.last))
	p.lastSample = now
	return p.last
}

// Start the current sample to now, such that the next call to Sample will be count the difference from the start of this method instead of the last call to Sample.
func (p *Period) Start() {
	p.lastSample = time.Now()
}

// LastPeriod returns the last period measurement.
func (p *Period) LastPeriod() time.Duration {
	return p.last
}

// Average returns the current moving period measured.
func (p *Period) Average() time.Duration {
	return p.average
}
