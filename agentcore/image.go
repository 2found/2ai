package agentcore

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/jpeg"
	"image/png"
	"math"

	xdraw "golang.org/x/image/draw"
	_ "golang.org/x/image/webp"
)

const (
	richImageMaxDimension     = 1568
	richImageMinDimension     = 200
	richImageTargetBytes      = 500 * 1024
	richImageMaxInputBytes    = 16 * 1024 * 1024
	richImageMaxDecodedPixels = 16 * 1024 * 1024
)

type normalizedRichImage struct {
	part                          ContentPart
	originalWidth, originalHeight int
	width, height                 int
	bytes                         int
}

// normalizeRichImage validates decoded image data, corrects caller MIME
// metadata, and makes the attachment portable across provider implementations.
// The policy follows OMP's proven vision envelope: longest edge <= 1568px,
// shortest edge >= 200px, and a 500KiB per-image target (further reduced by the
// current aggregate budget). It is pure Go so laptop and server paths are
// identical and do not require ImageMagick, sharp, or another sidecar binary.
func normalizeRichImage(part ContentPart, maxBytes int) (normalizedRichImage, bool) {
	if maxBytes <= 0 || part.Data == "" {
		return normalizedRichImage{}, false
	}
	maxEncoded := base64.StdEncoding.EncodedLen(richImageMaxInputBytes)
	if len(part.Data) > maxEncoded {
		return normalizedRichImage{}, false
	}
	decoded, err := base64.StdEncoding.DecodeString(part.Data)
	if err != nil || len(decoded) == 0 || len(decoded) > richImageMaxInputBytes {
		return normalizedRichImage{}, false
	}

	config, format, err := image.DecodeConfig(bytes.NewReader(decoded))
	if err != nil || config.Width <= 0 || config.Height <= 0 ||
		config.Width > richImageMaxDecodedPixels/config.Height {
		return normalizedRichImage{}, false
	}
	mime := richImageMIME(format)
	if mime == "" {
		return normalizedRichImage{}, false
	}
	source, decodedFormat, err := image.Decode(bytes.NewReader(decoded))
	if err != nil || richImageMIME(decodedFormat) != mime {
		return normalizedRichImage{}, false
	}

	part.Type = ContentPartImage
	part.Text = ""
	part.DataRef = ""
	part.MIMEType = mime
	budget := min(maxBytes, richImageTargetBytes)
	comfortable := budget / 4
	withinDimensions := config.Width >= richImageMinDimension && config.Height >= richImageMinDimension &&
		config.Width <= richImageMaxDimension && config.Height <= richImageMaxDimension
	// WebP is always converted to PNG/JPEG. Some otherwise OpenAI-compatible
	// local inference stacks cannot decode it, and AgentRay must remain portable
	// without a model-catalog exception for every such endpoint.
	if withinDimensions && len(decoded) <= comfortable && mime != "image/webp" {
		return normalizedRichImage{
			part: part, originalWidth: config.Width, originalHeight: config.Height,
			width: config.Width, height: config.Height, bytes: len(decoded),
		}, true
	}

	targetWidth, targetHeight := richImageTargetDimensions(config.Width, config.Height)
	result, width, height, ok := encodeRichImageWithin(source, targetWidth, targetHeight, budget)
	if !ok {
		return normalizedRichImage{}, false
	}
	part.Data = base64.StdEncoding.EncodeToString(result.data)
	part.MIMEType = result.mime
	return normalizedRichImage{
		part: part, originalWidth: config.Width, originalHeight: config.Height,
		width: width, height: height, bytes: len(result.data),
	}, true
}

func richImageMIME(format string) string {
	switch format {
	case "png":
		return "image/png"
	case "jpeg":
		return "image/jpeg"
	case "gif":
		return "image/gif"
	case "webp":
		return "image/webp"
	default:
		return ""
	}
}

func richImageTargetDimensions(width, height int) (int, int) {
	w, h := float64(width), float64(height)
	downscale := math.Min(1, math.Min(float64(richImageMaxDimension)/w, float64(richImageMaxDimension)/h))
	w, h = math.Max(1, math.Round(w*downscale)), math.Max(1, math.Round(h*downscale))
	if w < richImageMinDimension || h < richImageMinDimension {
		shortEdge := math.Min(w, h)
		upscale := math.Min(float64(richImageMinDimension)/shortEdge,
			math.Min(float64(richImageMaxDimension)/w, float64(richImageMaxDimension)/h))
		if upscale > 1 {
			w, h = math.Round(w*upscale), math.Round(h*upscale)
		}
		// An extreme aspect ratio cannot satisfy both the floor and cap by a
		// uniform scale. Match OMP's final fill behavior for that rare case.
		w = math.Min(richImageMaxDimension, math.Max(richImageMinDimension, w))
		h = math.Min(richImageMaxDimension, math.Max(richImageMinDimension, h))
	}
	return int(w), int(h)
}

type richImageEncoding struct {
	data []byte
	mime string
}

func encodeRichImageWithin(source image.Image, targetWidth, targetHeight, budget int) (richImageEncoding, int, int, bool) {
	qualitySteps := []int{70, 60, 50, 40}
	scaleSteps := []float64{1, 0.75, 0.5, 0.35, 0.25}
	var best richImageEncoding
	bestWidth, bestHeight := targetWidth, targetHeight
	for scaleIndex, scale := range scaleSteps {
		width := max(1, int(math.Round(float64(targetWidth)*scale)))
		height := max(1, int(math.Round(float64(targetHeight)*scale)))
		if scaleIndex > 0 && (width < 100 || height < 100) {
			break
		}
		scaled := scaleRichImage(source, width, height)
		if scaleIndex == 0 {
			pngCandidate, pngErr := encodeRichPNG(scaled)
			jpegCandidate, jpegErr := encodeRichJPEG(scaled, 80)
			if pngErr == nil {
				best, bestWidth, bestHeight = pngCandidate, width, height
			}
			if jpegErr == nil && (len(best.data) == 0 || len(jpegCandidate.data) < len(best.data)) {
				best, bestWidth, bestHeight = jpegCandidate, width, height
			}
			if len(best.data) > 0 && len(best.data) <= budget {
				return best, bestWidth, bestHeight, true
			}
		}
		for _, quality := range qualitySteps {
			encoded, err := encodeRichJPEG(scaled, quality)
			if err != nil {
				continue
			}
			if len(best.data) == 0 || len(encoded.data) < len(best.data) {
				best, bestWidth, bestHeight = encoded, width, height
			}
			if len(encoded.data) <= budget {
				return encoded, width, height, true
			}
		}
	}
	if len(best.data) > 0 && len(best.data) <= budget {
		return best, bestWidth, bestHeight, true
	}
	return richImageEncoding{}, 0, 0, false
}

func scaleRichImage(source image.Image, width, height int) *image.NRGBA {
	destination := image.NewNRGBA(image.Rect(0, 0, width, height))
	xdraw.CatmullRom.Scale(destination, destination.Bounds(), source, source.Bounds(), draw.Src, nil)
	return destination
}

func encodeRichPNG(source image.Image) (richImageEncoding, error) {
	var output bytes.Buffer
	encoder := png.Encoder{CompressionLevel: png.BestCompression}
	if err := encoder.Encode(&output, source); err != nil {
		return richImageEncoding{}, err
	}
	return richImageEncoding{data: output.Bytes(), mime: "image/png"}, nil
}

func encodeRichJPEG(source image.Image, quality int) (richImageEncoding, error) {
	// JPEG has no alpha channel. Composite onto white so transparent charts and
	// screenshots do not turn black when JPEG wins the size comparison.
	background := image.NewRGBA(source.Bounds())
	draw.Draw(background, background.Bounds(), &image.Uniform{C: color.White}, image.Point{}, draw.Src)
	draw.Draw(background, background.Bounds(), source, source.Bounds().Min, draw.Over)
	var output bytes.Buffer
	if err := jpeg.Encode(&output, background, &jpeg.Options{Quality: quality}); err != nil {
		return richImageEncoding{}, err
	}
	return richImageEncoding{data: output.Bytes(), mime: "image/jpeg"}, nil
}

func (image normalizedRichImage) dimensionNote() string {
	if image.originalWidth == image.width && image.originalHeight == image.height {
		return ""
	}
	xScale := float64(image.originalWidth) / float64(image.width)
	yScale := float64(image.originalHeight) / float64(image.height)
	if math.Abs(xScale-yScale) < 0.01 {
		return fmt.Sprintf("[Image normalized from %dx%d to %dx%d; multiply coordinates by %.2f to map to the original]",
			image.originalWidth, image.originalHeight, image.width, image.height, xScale)
	}
	return fmt.Sprintf("[Image normalized from %dx%d to %dx%d; multiply x by %.2f and y by %.2f to map to the original]",
		image.originalWidth, image.originalHeight, image.width, image.height, xScale, yScale)
}
