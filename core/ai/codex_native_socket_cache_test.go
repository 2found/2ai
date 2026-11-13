package ai

import (
	"context"
	"encoding/json"
	"os"
	"sort"
	"sync"
	"testing"
	"time"
)

func TestPiCodexSocketCacheOracle(t *testing.T) {
	raw, err := os.ReadFile("testdata/pi-codex-socket-cache.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		UpstreamCommit string
		Cases          []struct {
			Input struct {
				Name    string
				Actions []struct {
					Op, Lease, Session, Account string
					Keep, Fire                  bool
					Ms                          int64
					State                       *int
				}
			}
			Expected []json.RawMessage
		}
	}
	if err = json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != "eeac84ca92498ac18b6832754d01aef1d3c5f654" || len(fixture.Cases) != 16 {
		t.Fatal("unexpected cache oracle coverage")
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Input.Name, func(t *testing.T) {
			cache := newCodexSocketCache()
			now := time.Unix(0, 0)
			cache.now = func() time.Time { return now }
			type timer struct {
				due    time.Time
				run    func()
				active bool
			}
			var timers []*timer
			cache.after = func(d time.Duration, run func()) func() {
				v := &timer{now.Add(d), run, true}
				timers = append(timers, v)
				return func() { v.active = false }
			}
			leases := map[string]codexSocketLease{}
			names := []string{}
			ids := map[*codexSocket]int{}
			states := map[*codexSocket]*int{}
			closes := []any{}
			connect := func(context.Context) (*codexSocket, error) {
				s := &codexSocket{}
				id := len(ids)
				ids[s] = id
				state := 1
				states[s] = &state
				s.ready = func() *int { return states[s] }
				s.close = func(code int, reason string) {
					closes = append(closes, map[string]any{"socket": id, "code": code, "reason": reason})
					closed := 3
					states[s] = &closed
				}
				return s, nil
			}
			for i, step := range tc.Input.Actions {
				switch step.Op {
				case "acquire":
					l, e := cache.acquire(context.Background(), step.Session, step.Account, connect)
					if e != nil {
						t.Fatal(e)
					}
					leases[step.Lease] = l
					names = append(names, step.Lease)
				case "release":
					leases[step.Lease].release(step.Keep)
				case "state":
					states[leases[step.Lease].socket] = step.State
				case "close":
					cache.closeSessions(step.Session)
				case "time":
					now = now.Add(time.Duration(step.Ms) * time.Millisecond)
					if step.Fire {
						for _, v := range append([]*timer(nil), timers...) {
							if v.active && !v.due.After(now) {
								v.active = false
								v.run()
							}
						}
					}
				}
				ls := []any{}
				for _, name := range names {
					l := leases[name]
					ls = append(ls, map[string]any{"name": name, "socket": ids[l.socket], "reused": l.reused, "cached": l.entry != nil})
				}
				keys := []codexSocketKey{}
				for k := range cache.entries {
					keys = append(keys, k)
				}
				sort.Slice(keys, func(i, j int) bool { return keys[i].session+keys[i].account < keys[j].session+keys[j].account })
				entries := []any{}
				for _, k := range keys {
					e := cache.entries[k]
					entries = append(entries, map[string]any{"session": k.session, "account": k.account, "socket": ids[e.socket], "busy": e.busy})
				}
				assertPiJSON(t, tc.Expected[i], map[string]any{"leases": ls, "entries": entries, "closes": closes})
			}
		})
	}
}

func TestCodexSocketCacheConcurrentConnectionOwnership(t *testing.T) {
	cache := newCodexSocketCache()
	defer cache.closeSessions("")
	entered := make(chan struct{})
	resume := make(chan struct{})
	first := make(chan codexSocketLease, 1)
	go func() {
		l, _ := cache.acquire(context.Background(), "s", "a", func(context.Context) (*codexSocket, error) { close(entered); <-resume; return &codexSocket{}, nil })
		first <- l
	}()
	<-entered
	second, err := cache.acquire(context.Background(), "s", "a", func(context.Context) (*codexSocket, error) { return &codexSocket{}, nil })
	if err != nil {
		t.Fatal(err)
	}
	close(resume)
	owner := <-first
	// Source inserts after connection completes, so the last completed connection owns the key.
	second.release(false)
	cache.mu.Lock()
	entry := cache.entries[codexSocketKey{"s", "a"}]
	cache.mu.Unlock()
	if entry != owner.entry {
		t.Fatal("stale release removed the current connection")
	}
	var wg sync.WaitGroup
	for i := 0; i < 32; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			l, e := cache.acquire(context.Background(), "s", "a", func(context.Context) (*codexSocket, error) { return &codexSocket{}, nil })
			if e != nil {
				t.Error(e)
				return
			}
			if l.entry != nil {
				t.Error("busy connection was shared")
			}
			l.release(true)
		}()
	}
	wg.Wait()
	owner.release(false)
}
