package telemetry

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"math"
	"strconv"
	"sync"
	"sync/atomic"
	"time"
)

const SafeSchemaVersion = "2ai.safe.v1"
const SafeAccountingSchema = "2ai.disjoint-tokens.v1"
const SafeIdentifierLimit = 128
const SafeQueueLimit = 2048
const SafeRecordLimit = 4096
const SafeDrainLimit = 256

type SafeCategory string

const (
	CategoryMain       SafeCategory = "main"
	CategoryCompaction SafeCategory = "compaction"
	CategoryAdvisor    SafeCategory = "advisor"
	CategoryMemory     SafeCategory = "memory"
	CategoryChild      SafeCategory = "child"
)

// SafeLabels can only be constructed by an explicit host approval. Approval
// must use configured public IDs, never provider response text or raw metadata.
// The zero value represents unavailable labels. Values are immutable.
type SafeLabels struct{ provider, model, tool string }

func ApproveModelLabels(provider, model string) (SafeLabels, error) {
	if !safeIdentifier(provider) || !safeIdentifier(model) {
		return SafeLabels{}, errors.New("invalid safe model labels")
	}
	return SafeLabels{provider: provider, model: model}, nil
}
func ApproveToolLabel(tool string) (SafeLabels, error) {
	if !safeIdentifier(tool) {
		return SafeLabels{}, errors.New("invalid safe tool label")
	}
	return SafeLabels{tool: tool}, nil
}
func safeIdentifier(s string) bool {
	if len(s) == 0 || len(s) > SafeIdentifierLimit {
		return false
	}
	for i := range len(s) {
		c := s[i]
		if !((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.') {
			return false
		}
	}
	return true
}

type SafeTokens struct {
	Input      float64 `json:"input"`
	Output     float64 `json:"output"`
	CacheRead  float64 `json:"cache_read"`
	CacheWrite float64 `json:"cache_write"`
}

// SafeAccounting is evidence, not a billing claim. Unknown values are null.
// Only attempt_settled records are additive. Reconciliation is comparison-only.
type SafeAccounting struct {
	UsageObserved   bool        `json:"usage_observed"`
	PricingObserved bool        `json:"pricing_observed"`
	UsageSource     string      `json:"usage_source"`
	PricingSource   string      `json:"pricing_source"`
	PriceState      string      `json:"price_state"`
	Complete        bool        `json:"complete"`
	AttemptCoverage string      `json:"attempt_coverage"`
	Tokens          *SafeTokens `json:"tokens"`
	KnownCostUSD    *float64    `json:"known_cost_usd"`
}

// SafeRecord has no private attributes, errors, messages or serializer hooks.
// Every optional ID is null when unavailable. All records are detached values.
type SafeRecord struct {
	SchemaVersion     string         `json:"schema_version"`
	AccountingSchema  string         `json:"accounting_schema"`
	Kind              string         `json:"kind"`
	ProducerEventID   string         `json:"producer_event_id"`
	ExecutionID       string         `json:"execution_id"`
	OriginExecutionID *string        `json:"origin_execution_id"`
	SpanID            *string        `json:"span_id"`
	ParentSpanID      *string        `json:"parent_span_id"`
	ParentExecutionID *string        `json:"parent_execution_id"`
	CallID            *string        `json:"call_id"`
	AttemptID         *string        `json:"attempt_id"`
	AttemptOrdinal    int            `json:"attempt_ordinal"`
	ProducerAdmitted  bool           `json:"producer_admitted"`
	Category          SafeCategory   `json:"category"`
	SpanKind          string         `json:"span_kind"`
	ProviderID        *string        `json:"provider_id"`
	ModelID           *string        `json:"model_id"`
	ToolID            *string        `json:"tool_id"`
	LabelsState       string         `json:"labels_state"`
	StartedAt         time.Time      `json:"started_at"`
	SettledAt         time.Time      `json:"settled_at"`
	Duration          time.Duration  `json:"duration_ns"`
	Status            string         `json:"status"`
	FailureKind       string         `json:"failure_kind"`
	Accounting        SafeAccounting `json:"accounting"`
	Reconciliation    *SafeAggregate `json:"reconciliation"`
}

// SafeAggregate preserves existing reported totals for comparison only. The
// producer cannot infer wire usage or known pricing from these host totals.
type SafeAggregate struct {
	ReportedTokens  SafeTokens `json:"reported_tokens"`
	ReportedCostUSD float64    `json:"reported_cost_usd"`
	Additive        bool       `json:"additive"`
}

type SafeObserverOptions struct{ Capacity, RecordBytes int }
type SafeHealth struct {
	Capacity        int    `json:"capacity"`
	Queued          int    `json:"queued"`
	Admitted        uint64 `json:"admitted"`
	Overflow        uint64 `json:"overflow"`
	Oversize        uint64 `json:"oversize"`
	ClosedAdmission uint64 `json:"closed_admission"`
	Contended       uint64 `json:"contended"`
	Closed          bool   `json:"closed"`
	Complete        bool   `json:"complete"`
}

// SafeObserver is a finite passive pull queue. It owns no goroutine and invokes
// no sink. The host drains, persists, retries and deduplicates detached records.
// Intake lifetime belongs to the host, independently of any root settlement.
type SafeObserver struct {
	queue                                                    chan SafeRecord
	maxBytes                                                 int
	prefix                                                   string
	sequence                                                 atomic.Uint64
	mu                                                       sync.RWMutex
	closed                                                   bool
	admitted, overflow, oversize, closedAdmission, contended atomic.Uint64
}

func NewSafeObserver(options SafeObserverOptions) (*SafeObserver, error) {
	if options.Capacity == 0 {
		options.Capacity = SafeQueueLimit
	}
	if options.RecordBytes == 0 {
		options.RecordBytes = SafeRecordLimit
	}
	if options.Capacity < 1 || options.Capacity > SafeQueueLimit || options.RecordBytes < 1 || options.RecordBytes > SafeRecordLimit {
		return nil, errors.New("invalid safe observer bounds")
	}
	var nonce [16]byte
	if _, err := rand.Read(nonce[:]); err != nil {
		return nil, err
	}
	return &SafeObserver{queue: make(chan SafeRecord, options.Capacity), maxBytes: options.RecordBytes, prefix: hex.EncodeToString(nonce[:])}, nil
}
func (o *SafeObserver) id() string { return o.prefix + "-" + strconv.FormatUint(o.sequence.Add(1), 10) }
func (o *SafeObserver) admit(r SafeRecord) {
	// All fields are already bounded scalars. Serialization cannot call host code.
	raw, err := json.Marshal(r)
	if err != nil || len(raw) > o.maxBytes {
		o.oversize.Add(1)
		return
	}
	if !o.mu.TryRLock() {
		o.contended.Add(1)
		return
	}
	defer o.mu.RUnlock()
	if o.closed {
		o.closedAdmission.Add(1)
		return
	}
	select {
	case o.queue <- r:
		o.admitted.Add(1)
	default:
		o.overflow.Add(1)
	}
}

// Drain consumes at most min(limit, 256) records. A nonpositive limit consumes
// none. Host delivery retries retain the returned event IDs; there is no replay.
func (o *SafeObserver) Drain(limit int) []SafeRecord {
	if o == nil || limit <= 0 {
		return nil
	}
	limit = min(limit, SafeDrainLimit)
	result := make([]SafeRecord, 0, min(limit, len(o.queue)))
	for range limit {
		select {
		case r := <-o.queue:
			result = append(result, r)
		default:
			return result
		}
	}
	return result
}
func (o *SafeObserver) Close() {
	if o != nil {
		o.mu.Lock()
		o.closed = true
		o.mu.Unlock()
	}
}
func (o *SafeObserver) Health() SafeHealth {
	if o == nil {
		return SafeHealth{}
	}
	o.mu.RLock()
	closed := o.closed
	o.mu.RUnlock()
	h := SafeHealth{Capacity: cap(o.queue), Queued: len(o.queue), Admitted: o.admitted.Load(), Overflow: o.overflow.Load(), Oversize: o.oversize.Load(), ClosedAdmission: o.closedAdmission.Load(), Contended: o.contended.Load(), Closed: closed}
	h.Complete = h.Overflow == 0 && h.Oversize == 0 && h.ClosedAdmission == 0 && h.Contended == 0
	return h
}

type safeKey struct{}
type safeScope struct {
	observer                                             *SafeObserver
	execution, origin, span, parentSpan, parentExecution string
	category                                             SafeCategory
}
type SafeExecution struct {
	ExecutionID, OriginExecutionID string
	Category                       SafeCategory
}

// WithSafeObserver accepts opaque host-approved correlation only. Runtime,
// tenant, config and agent identity remain the consuming host's responsibility.
func WithSafeObserver(ctx context.Context, observer *SafeObserver, execution SafeExecution) (context.Context, error) {
	if observer == nil {
		return ctx, nil
	}
	if !safeIdentifier(execution.ExecutionID) || (execution.OriginExecutionID != "" && !safeIdentifier(execution.OriginExecutionID)) {
		return ctx, errors.New("invalid safe execution correlation")
	}
	return context.WithValue(ctx, safeKey{}, &safeScope{observer: observer, execution: execution.ExecutionID, origin: execution.OriginExecutionID, category: safeCategory(execution.Category)}), nil
}
func safeFrom(ctx context.Context) *safeScope { s, _ := ctx.Value(safeKey{}).(*safeScope); return s }
func safeCategory(c SafeCategory) SafeCategory {
	switch c {
	case CategoryMain, CategoryCompaction, CategoryAdvisor, CategoryMemory, CategoryChild:
		return c
	}
	return CategoryMain
}
func WithSafeCategory(ctx context.Context, category SafeCategory) context.Context {
	if s := safeFrom(ctx); s != nil {
		copied := *s
		copied.category = safeCategory(category)
		return context.WithValue(ctx, safeKey{}, &copied)
	}
	return ctx
}

// SafeOrigin carries only immutable safe observation metadata across host work
// queues. It carries no cancellation, private recorder or application context.
type SafeOrigin struct {
	scope *safeScope
	known bool
}

func CaptureSafeOrigin(ctx context.Context) SafeOrigin {
	s := safeFrom(ctx)
	return SafeOrigin{scope: s, known: s != nil}
}
func (o SafeOrigin) Merge(other SafeOrigin) SafeOrigin {
	if o.scope == nil {
		o.scope = other.scope
		o.known = false
		return o
	}
	if other.scope == nil || o.scope.observer != other.scope.observer || o.scope.execution != other.scope.execution {
		o.known = false
	}
	return o
}
func BindSafeOrigin(ctx context.Context, origin SafeOrigin, category SafeCategory) context.Context {
	if origin.scope == nil {
		return ctx
	}
	s := &safeScope{observer: origin.scope.observer, execution: origin.scope.observer.id(), category: safeCategory(category)}
	if origin.known {
		s.origin = origin.scope.execution
		s.parentExecution = origin.scope.execution
		s.parentSpan = origin.scope.span
	}
	return context.WithValue(ctx, safeKey{}, s)
}
func DetachSafeExecution(ctx context.Context, category SafeCategory) context.Context {
	return BindSafeOrigin(ctx, CaptureSafeOrigin(ctx), category)
}

// WithoutSafeOrigin retains the independent execution while acknowledging that
// coalesced durable work has no trustworthy single originating execution.
func WithoutSafeOrigin(ctx context.Context) context.Context {
	if s := safeFrom(ctx); s != nil {
		copied := *s
		copied.origin, copied.parentSpan, copied.parentExecution = "", "", ""
		return context.WithValue(ctx, safeKey{}, &copied)
	}
	return ctx
}

// SafeContext composes safe live settlement with the existing private backend.
// It also works when that backend is inert or refuses late children.
func SafeContext(private Context, ctx context.Context) Context {
	private.safe = safeFrom(ctx)
	return private
}
func ptrID(value string) *string {
	if value == "" {
		return nil
	}
	copy := value
	return &copy
}
func (s *safeScope) record(kind string, started time.Time) SafeRecord {
	return SafeRecord{SchemaVersion: SafeSchemaVersion, AccountingSchema: SafeAccountingSchema, Kind: kind, ProducerEventID: s.observer.id(), ExecutionID: s.execution, OriginExecutionID: ptrID(s.origin), SpanID: ptrID(s.span), ParentSpanID: ptrID(s.parentSpan), ParentExecutionID: ptrID(s.parentExecution), Category: s.category, SpanKind: "unknown", LabelsState: "unavailable", StartedAt: started.Round(0).UTC(), SettledAt: time.Now().UTC(), Duration: time.Since(started), Status: "unknown", FailureKind: "unknown", Accounting: NormalizeSafeAccounting(SafeAccounting{})}
}

// NewSafeCall allocates one logical call identity, shared by every retry/rung.
type SafeCall struct {
	scope *safeScope
	id    string
}

func NewSafeCall(ctx context.Context) SafeCall {
	if s := safeFrom(ctx); s != nil {
		return SafeCall{scope: s, id: s.observer.id()}
	}
	return SafeCall{}
}

type SafeAttempt struct {
	scope   *safeScope
	record  SafeRecord
	started time.Time
	once    sync.Once
}

func (c SafeCall) StartAttempt(ordinal int, labels SafeLabels) *SafeAttempt {
	if c.scope == nil {
		return nil
	}
	started := time.Now()
	r := c.scope.record("attempt_settled", started)
	r.CallID = ptrID(c.id)
	r.AttemptID = ptrID(c.scope.observer.id())
	r.AttemptOrdinal = ordinal
	r.SpanKind = "model"
	r.ProviderID = ptrID(labels.provider)
	r.ModelID = ptrID(labels.model)
	if r.ProviderID != nil && r.ModelID != nil {
		r.LabelsState = "approved"
	}
	return &SafeAttempt{scope: c.scope, record: r, started: started}
}
func (a *SafeAttempt) Finish(status, failureKind string, accounting SafeAccounting) {
	a.FinishModel(status, failureKind, accounting, "", "")
}

// FinishModel checks response selection against the immutable approved binding.
// An unapproved internal fallback is unavailable; response text is never copied
// into a label. Empty response selection retains the host candidate binding.
func (a *SafeAttempt) FinishModel(status, failureKind string, accounting SafeAccounting, provider, model string) {
	a.FinishAdmission(status, failureKind, accounting, provider, model, true)
}

// FinishAdmission distinguishes an attempted open from an admitted producer.
// An admission failure remains an attempt with unavailable provider usage.
func (a *SafeAttempt) FinishAdmission(status, failureKind string, accounting SafeAccounting, provider, model string, admitted bool) {
	if a == nil {
		return
	}
	a.once.Do(func() {
		r := a.record
		r.ProducerAdmitted = admitted
		if (provider != "" && (r.ProviderID == nil || provider != *r.ProviderID)) || (model != "" && (r.ModelID == nil || model != *r.ModelID)) {
			r.ProviderID, r.ModelID, r.LabelsState = nil, nil, "unavailable"
		}
		r.SettledAt = time.Now().UTC()
		r.Duration = time.Since(a.started)
		r.Status = safeStatus(status)
		r.FailureKind = SafeFailureKind(failureKind)
		if !admitted {
			accounting = SafeAccounting{}
		}
		accounting.AttemptCoverage = "observed"
		r.Accounting = NormalizeSafeAccounting(accounting)
		a.scope.observer.admit(r)
	})
}
func safeStatus(s string) string {
	switch s {
	case "completed", "stopped", "cancelled", "failed":
		return s
	}
	return "unknown"
}
func SafeFailureKind(s string) string {
	switch s {
	case "none", "host_callback", "host_preparation", "request_cancelled", "request_timeout", "provider_transport", "provider_auth", "provider_rate_limit", "provider_unavailable", "provider_rejected", "provider_protocol", "provider_decode", "tool_denied", "unknown":
		return s
	}
	return "unknown"
}
func finiteNonnegative(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) && v >= 0 }
func NormalizeSafeAccounting(a SafeAccounting) SafeAccounting {
	if a.UsageSource != "wire" && a.UsageSource != "explicit" && a.UsageSource != "invalid" {
		a.UsageSource = "unknown"
		a.UsageObserved = false
	}
	if a.PricingSource != "model" && a.PricingSource != "explicit" && a.PricingSource != "invalid" {
		a.PricingSource = "unknown"
		a.PricingObserved = false
	}
	if a.UsageSource == "invalid" {
		a.UsageObserved = false
	}
	if a.PricingSource == "invalid" {
		a.PricingObserved = false
	}
	if a.AttemptCoverage != "observed" && a.AttemptCoverage != "unavailable" {
		a.AttemptCoverage = "unavailable"
	}
	if a.Tokens != nil {
		t := *a.Tokens
		a.Tokens = &t
		if !finiteNonnegative(t.Input) || !finiteNonnegative(t.Output) || !finiteNonnegative(t.CacheRead) || !finiteNonnegative(t.CacheWrite) {
			a.UsageObserved = false
			a.UsageSource = "invalid"
		}
	}
	if a.Tokens == nil {
		a.UsageObserved = false
	}
	if !a.UsageObserved {
		a.Tokens = nil
	}
	if a.KnownCostUSD != nil {
		cost := *a.KnownCostUSD
		a.KnownCostUSD = &cost
		if !finiteNonnegative(cost) {
			a.PricingObserved = false
			a.PricingSource = "invalid"
		}
	}
	if a.KnownCostUSD == nil {
		a.PricingObserved = false
	}
	if !a.PricingObserved || !a.UsageObserved {
		a.KnownCostUSD = nil
	}
	a.PriceState = "unknown"
	if a.UsageObserved {
		a.PriceState = "unpriced"
		if a.PricingObserved && a.KnownCostUSD != nil {
			a.PriceState = "priced"
		}
	}
	a.Complete = a.UsageObserved && a.PriceState == "priced" && a.AttemptCoverage == "observed"
	return a
}

// ReconcileSafe reports aggregates without inventing physical attempt coverage.
func ReconcileSafe(ctx context.Context, accounting SafeAccounting) {
	if s := safeFrom(ctx); s != nil {
		r := s.record("reconciliation", time.Now())
		if accounting.Tokens != nil && accounting.KnownCostUSD != nil {
			t, cost := *accounting.Tokens, *accounting.KnownCostUSD
			if finiteNonnegative(t.Input) && finiteNonnegative(t.Output) && finiteNonnegative(t.CacheRead) && finiteNonnegative(t.CacheWrite) && finiteNonnegative(cost) {
				r.Reconciliation = &SafeAggregate{ReportedTokens: t, ReportedCostUSD: cost}
			}
		}
		accounting.AttemptCoverage = "unavailable"
		r.Accounting = NormalizeSafeAccounting(accounting)
		s.observer.admit(r)
	}
}
