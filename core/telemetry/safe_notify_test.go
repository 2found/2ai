package telemetry

import "testing"

func TestSafeObserverChangesCoalesceAndReportLoss(t *testing.T) {
	o, err := NewSafeObserver(SafeObserverOptions{Capacity: 1})
	if err != nil {
		t.Fatal(err)
	}
	select {
	case <-o.Changes():
		t.Fatal("idle observer signaled")
	default:
	}
	o.admit(SafeRecord{Kind: "one"})
	o.admit(SafeRecord{Kind: "overflow"})
	select {
	case <-o.Changes():
	default:
		t.Fatal("producer did not signal")
	}
	select {
	case <-o.Changes():
		t.Fatal("notifications did not coalesce")
	default:
	}
	if h := o.Health(); h.Queued != 1 || h.Overflow != 1 {
		t.Fatal(h)
	}
	if rows := o.Drain(256); len(rows) != 1 || rows[0].Kind != "one" {
		t.Fatal(rows)
	}
	o.Close()
	select {
	case <-o.Changes():
	default:
		t.Fatal("close did not signal")
	}
	select {
	case <-o.Changes():
		t.Fatal("closed channel would spin collector")
	default:
	}
	o.admit(SafeRecord{})
	select {
	case <-o.Changes():
	default:
		t.Fatal("lost admission did not signal")
	}
	if o.Health().ClosedAdmission != 1 {
		t.Fatal(o.Health())
	}
	oversize, _ := NewSafeObserver(SafeObserverOptions{RecordBytes: 1})
	oversize.admit(SafeRecord{})
	select {
	case <-oversize.Changes():
	default:
		t.Fatal("oversize loss did not signal")
	}
}

func BenchmarkSafeObserverCreate(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		o, err := NewSafeObserver(SafeObserverOptions{Capacity: 1792})
		if err != nil {
			b.Fatal(err)
		}
		o.Close()
	}
}

func BenchmarkSafeObserverAdmission(b *testing.B) {
	o, err := NewSafeObserver(SafeObserverOptions{Capacity: 1792})
	if err != nil {
		b.Fatal(err)
	}
	defer o.Close()
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		o.admit(SafeRecord{Kind: "attempt_settled"})
		if rows := o.Drain(1); len(rows) != 1 {
			b.Fatal("record lost")
		}
	}
}
