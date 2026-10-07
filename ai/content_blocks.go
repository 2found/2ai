package ai

import (
	"encoding/json"

	"github.com/2found/2ai/jsonjs"
)

// BlockList retains content-array identity across shallow message copies. A nil
// entry is a hole; NullContentBlock represents an explicit null. Its typed view
// uses the shared JavaScript array implementation for length and sparse slots.
// Callers synchronize live stream reads and writes through the owning stream.
type BlockList struct {
	entries jsonjs.Array
}

func NewBlockList(blocks ...*ContentBlock) *BlockList {
	list := &BlockList{}
	list.Append(blocks...)
	return list
}

func (l *BlockList) Len() int {
	if l == nil {
		return 0
	}
	return l.entries.Len()
}

func (l *BlockList) Get(index int) *ContentBlock {
	if l == nil {
		return nil
	}
	block, _ := l.entries.Get(index).(*ContentBlock)
	return block
}

func (l *BlockList) Set(index int, block *ContentBlock) {
	l.entries.Set(index, block)
	if block == nil {
		l.entries.Delete(index)
	}
}

func (l *BlockList) Delete(index int)     { l.entries.Delete(index) }
func (l *BlockList) SetLength(length int) { l.entries.SetLength(length) }

func (l *BlockList) Append(blocks ...*ContentBlock) int {
	for _, block := range blocks {
		l.Set(l.Len(), block)
	}
	return l.Len()
}

func (l *BlockList) Pop() *ContentBlock {
	block, _ := l.entries.Pop().(*ContentBlock)
	return block
}

// Values returns a detached outer slice; the block objects remain shared.
// Mutate membership through Set/Append/Delete/SetLength, not this projection.
func (l *BlockList) Values() []*ContentBlock {
	blocks := make([]*ContentBlock, l.Len())
	for i := range blocks {
		blocks[i] = l.Get(i)
	}
	return blocks
}

func (l *BlockList) MarshalJSON() ([]byte, error) { return json.Marshal(l.Values()) }

func (l *BlockList) UnmarshalJSON(data []byte) error {
	var blocks []*ContentBlock
	if err := json.Unmarshal(data, &blocks); err != nil {
		return err
	}
	for i, block := range blocks {
		if block == nil {
			blocks[i] = NullContentBlock()
		}
	}
	l.entries = jsonjs.Array{}
	l.Append(blocks...)
	return nil
}
