package ai

func boundedFailureKind(kind string) string {
	switch kind {
	case "none", "host_callback", "host_preparation", "request_cancelled", "request_timeout", "provider_transport", "provider_auth", "provider_rate_limit", "provider_unavailable", "provider_rejected", "provider_protocol", "provider_decode", "unknown":
		return kind
	}
	return "unknown"
}
