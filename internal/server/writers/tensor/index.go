package tensor

type indexer struct {
	lastUpdate uint
}

func (idx *indexer) updateIndex(last uint) {
	idx.lastUpdate = last
}

func (idx *indexer) lastUpdateIndex() uint {
	return idx.lastUpdate
}
