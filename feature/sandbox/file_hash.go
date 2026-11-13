package sandbox

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strings"
)

// fileHashHexLen is a 64-bit truncated SHA-256 tag. It is compact enough to
// repeat in model tool calls while making accidental collisions negligible for
// workspace edit histories. The hash is over the file's raw bytes, so line
// endings and BOMs are part of the snapshot the edit is promising to replace.
const fileHashHexLen = 16

func fileContentHash(data []byte) string {
	sum := sha256.Sum256(data)
	return strings.ToUpper(hex.EncodeToString(sum[:fileHashHexLen/2]))
}

func normalizeFileHash(raw string) (string, error) {
	h := strings.ToUpper(strings.TrimSpace(raw))
	if len(h) != fileHashHexLen {
		return "", fmt.Errorf("must be %d hexadecimal characters", fileHashHexLen)
	}
	if _, err := hex.DecodeString(h); err != nil {
		return "", fmt.Errorf("must be %d hexadecimal characters", fileHashHexLen)
	}
	return h, nil
}
