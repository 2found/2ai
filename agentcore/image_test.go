package agentcore

import (
	"bytes"
	"encoding/base64"
	"image"
	"image/color"
	"image/png"
	"strings"
	"testing"
)

func TestNormalizeRichImageUpscalesDegenerateImage(t *testing.T) {
	got, ok := normalizeRichImage(ContentPart{
		Type: ContentPartImage, MIMEType: "image/png", Data: testRichPNG,
	}, richImageTargetBytes)
	if !ok {
		t.Fatal("valid 1x1 PNG was rejected")
	}
	decoded, err := base64.StdEncoding.DecodeString(got.part.Data)
	if err != nil {
		t.Fatal(err)
	}
	config, format, err := image.DecodeConfig(bytes.NewReader(decoded))
	if err != nil || format != "png" || config.Width != 200 || config.Height != 200 {
		t.Fatalf("normalized config = %+v format=%q err=%v", config, format, err)
	}
	if note := got.dimensionNote(); !strings.Contains(note, "1x1 to 200x200") || !strings.Contains(note, "coordinates") {
		t.Fatalf("dimension note = %q", note)
	}
}

func TestNormalizeRichImageCapsDimensions(t *testing.T) {
	data := testPatternPNG(t, 2000, 1000, false)
	got, ok := normalizeRichImage(ContentPart{
		Type: ContentPartImage, MIMEType: "image/png", Data: data,
	}, richImageTargetBytes)
	if !ok {
		t.Fatal("large valid PNG was rejected")
	}
	if got.width != 1568 || got.height != 784 || got.originalWidth != 2000 || got.originalHeight != 1000 {
		t.Fatalf("dimensions = original %dx%d normalized %dx%d", got.originalWidth, got.originalHeight, got.width, got.height)
	}
}

func TestNormalizeRichImageRecompressesToBudget(t *testing.T) {
	data := testPatternPNG(t, 1024, 1024, true)
	got, ok := normalizeRichImage(ContentPart{
		Type: ContentPartImage, MIMEType: "image/png", Data: data,
	}, richImageTargetBytes)
	if !ok {
		t.Fatal("compressible in-budget image was rejected")
	}
	decoded, err := base64.StdEncoding.DecodeString(got.part.Data)
	if err != nil {
		t.Fatal(err)
	}
	if len(decoded) > richImageTargetBytes {
		t.Fatalf("normalized bytes = %d, limit %d", len(decoded), richImageTargetBytes)
	}
	if got.part.MIMEType != "image/jpeg" {
		t.Fatalf("high-entropy image MIME = %q, want compact JPEG", got.part.MIMEType)
	}
}

func TestNormalizeRichImageTrustsDecodedFormatOverCallerMetadata(t *testing.T) {
	got, ok := normalizeRichImage(ContentPart{
		Type: ContentPartImage, MIMEType: "image/jpeg", Data: testPatternPNG(t, 256, 256, false),
	}, richImageTargetBytes)
	if !ok || got.part.MIMEType != "image/png" {
		t.Fatalf("decoded MIME was not corrected: ok=%v part=%+v", ok, got.part)
	}
}

func TestNormalizeRichImageRejectsAggregateBudgetItCannotMeet(t *testing.T) {
	_, ok := normalizeRichImage(ContentPart{
		Type: ContentPartImage, MIMEType: "image/png", Data: testPatternPNG(t, 256, 256, true),
	}, 1)
	if ok {
		t.Fatal("image unexpectedly fit a one-byte aggregate budget")
	}
}

func testPatternPNG(t *testing.T, width, height int, noisy bool) string {
	t.Helper()
	img := image.NewNRGBA(image.Rect(0, 0, width, height))
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			if noisy {
				// Deterministic high-frequency content that defeats lossless PNG
				// compression but remains cheap and reproducible in tests.
				v := uint32(x*73856093) ^ uint32(y*19349663) ^ uint32((x+y)*83492791)
				img.SetNRGBA(x, y, color.NRGBA{R: byte(v), G: byte(v >> 8), B: byte(v >> 16), A: 255})
			} else {
				img.SetNRGBA(x, y, color.NRGBA{R: byte(x), G: byte(y), B: 180, A: 255})
			}
		}
	}
	var output bytes.Buffer
	if err := png.Encode(&output, img); err != nil {
		t.Fatal(err)
	}
	return base64.StdEncoding.EncodeToString(output.Bytes())
}
