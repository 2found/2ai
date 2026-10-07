package sandbox

import (
	"strings"
	"testing"
)

func TestFileContentHashIsCompactDeterministicAndByteExact(t *testing.T) {
	a := fileContentHash([]byte("line\n"))
	if len(a) != fileHashHexLen || a != strings.ToUpper(a) {
		t.Fatalf("hash = %q, want %d uppercase hex characters", a, fileHashHexLen)
	}
	if again := fileContentHash([]byte("line\n")); again != a {
		t.Fatalf("same bytes hashed to %q then %q", a, again)
	}
	if crlf := fileContentHash([]byte("line\r\n")); crlf == a {
		t.Fatal("LF and CRLF snapshots collided; raw line endings must be guarded")
	}
	if bom := fileContentHash([]byte("\uFEFFline\n")); bom == a {
		t.Fatal("BOM and non-BOM snapshots collided; raw BOM bytes must be guarded")
	}
}

func TestNormalizeFileHashAcceptsCaseButRejectsMalformedValues(t *testing.T) {
	want := fileContentHash([]byte("x"))
	got, err := normalizeFileHash(strings.ToLower(want))
	if err != nil || got != want {
		t.Fatalf("normalize lowercase = %q, %v; want %q", got, err, want)
	}
	for _, bad := range []string{"", "ABC", "GGGGGGGGGGGGGGGG", want + "00"} {
		if _, err := normalizeFileHash(bad); err == nil {
			t.Errorf("normalizeFileHash(%q) succeeded", bad)
		}
	}
}
