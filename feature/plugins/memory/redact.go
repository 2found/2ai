package memory

import "regexp"

// RedactSecrets removes common credential-shaped strings before memory storage
// and auxiliary model calls. It deliberately does not claim general PII
// detection: names and preferences can be legitimate, consented user memory.
// Go's RE2 implementation keeps matching linear in input size.
var memorySecrets = []*regexp.Regexp{
	regexp.MustCompile(`(?i)-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----[\s\S]*?-----END (?:RSA |EC |OPENSSH )?PRIVATE KEY-----`),
	regexp.MustCompile(`(?:AKIA|ASIA)[A-Z0-9]{16}`),
	regexp.MustCompile(`(?:gh[pousr]_|github_pat_|npm_|xox[baprs]-|AIza|sk-(?:proj-|ant-)?)[A-Za-z0-9_-]{16,}`),
	regexp.MustCompile(`[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}`),
	regexp.MustCompile(`(?i)(?:bearer|basic)\s+[A-Za-z0-9_+/=.-]{8,}`),
	regexp.MustCompile(`(?i)(?:password|passwd|api[_-]?key|access[_-]?token|refresh[_-]?token|client[_-]?secret|secret)["']?\s*[:=]\s*["']?[^\s,"'{}]{4,}`),
	regexp.MustCompile(`(?i)\b(?:https?|postgres(?:ql)?|mysql|redis)://[^\s/@:]+:[^\s/@]+@`),
}

func RedactSecrets(text string) string {
	for _, pattern := range memorySecrets {
		text = pattern.ReplaceAllString(text, "[REDACTED]")
	}
	return text
}
