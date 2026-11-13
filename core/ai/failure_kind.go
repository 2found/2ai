package ai

// boundedFailureKind is the closed set of ai.failure_kind / error.type
// attribute values. "provider_admission" is a stream that failed to open for
// an untyped reason; the safe projection reports it as "unknown" with
// ProducerAdmitted false.
func boundedFailureKind(kind string) string {
	switch kind {
	case "none", "host_callback", "host_preparation", "request_cancelled", "request_timeout", "provider_transport", "provider_auth", "provider_rate_limit", "provider_unavailable", "provider_rejected", "provider_protocol", "provider_decode", "provider_admission", "unknown":
		return kind
	}
	return "unknown"
}
