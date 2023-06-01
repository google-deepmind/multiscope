// Copyright 2023 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
