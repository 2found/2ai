package ai

import (
	"fmt"
	"strings"

	"github.com/2found/2ai/ai/protocol"
)

// messageText joins the compatibility text with explicit text parts. RichTool
// currently places ordinary text in Message.Content, but accepting text parts
// here keeps the neutral message contract complete for host-injected callers.
func messageText(message protocol.Message) string {
	parts := make([]string, 0, 1+len(message.ContentParts))
	if message.Content != "" {
		parts = append(parts, message.Content)
	}
	for _, part := range message.ContentParts {
		if part.Type == protocol.ContentPartText && part.Text != "" {
			parts = append(parts, part.Text)
		}
	}
	return strings.Join(parts, "\n")
}

func messageImages(message protocol.Message) []protocol.ContentPart {
	images := make([]protocol.ContentPart, 0, len(message.ContentParts))
	for _, part := range message.ContentParts {
		if part.Type != protocol.ContentPartImage || part.Data == "" {
			continue
		}
		switch part.MIMEType {
		case "image/png", "image/jpeg", "image/gif", "image/webp":
			images = append(images, part)
		}
	}
	return images
}

func imageInputAllowed(caps protocol.ModelCapabilities) bool {
	return caps.ImageInput != protocol.CapabilityUnsupported
}

func textWithImageNotice(text string, count int, attached bool) string {
	if count == 0 {
		return text
	}
	note := fmt.Sprintf("[%d image output(s) omitted: provider/model path does not support image input]", count)
	if attached {
		note = fmt.Sprintf("[%d image output(s) attached]", count)
	}
	if text == "" {
		return note
	}
	return text + "\n" + note
}

func imageDataURL(part protocol.ContentPart) string {
	return "data:" + part.MIMEType + ";base64," + part.Data
}

func normalizedImageDetail(detail string, allowOriginal bool) string {
	switch detail {
	case "low", "high", "auto":
		return detail
	case "original":
		if allowOriginal {
			return detail
		}
	}
	return "auto"
}
