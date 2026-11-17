// Command modelcatalog compares 2ai's model tables (core/ai/window.go,
// core/ai/model_efforts.go) with oh-my-pi's model catalog and prints every
// difference worth a decision. It changes nothing; the model-catalog skill
// (.claude/skills/model-catalog) says how to act on what it prints.
//
//	go run ./test/modelcatalog                  # fetch the catalog from GitHub
//	go run ./test/modelcatalog -catalog models.json
//	go run ./test/modelcatalog -dump out.json   # also write the decoded catalog
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"slices"
	"sort"
	"strings"
	"time"

	"github.com/2found/2ai/core/ai"
)

const defaultCatalog = "https://raw.githubusercontent.com/can1357/oh-my-pi/main/packages/catalog/src/models.json"

// vendors are the catalog providers 2ai serves, by the 2ai vendor that serves
// them. A Claude Code login has no catalog entry of its own: it serves the
// anthropic models.
var vendors = map[string][]string{
	"anthropic":          {"anthropic", ai.VendorClaudeCode},
	"openai":             {"openai"},
	"openai-codex":       {ai.VendorOpenAICodex},
	"google":             {"google"},
	"google-antigravity": {ai.VendorGoogleAntigravity},
	"xai-oauth":          {ai.VendorXaiOAuth},
	"deepseek":           {"deepseek"},
	"devin":              {ai.VendorDevin},
	"opencode-go":        {"opencode-go"},
	"opencode-zen":       {"opencode-zen"},
}

// makers serve a family 2ai reaches through a router; their levels and
// windows are the ones the tables carry.
var makers = []string{"anthropic", "openai", "google", "xai", "deepseek", "moonshot", "zai", "minimax", "alibaba-coding-plan"}

type model struct {
	Provider, ID string
	Window       int
	Efforts      []string
	Reasons      bool
	Threshold    int
}

func main() {
	src := flag.String("catalog", defaultCatalog, "packed catalog: a URL or a file")
	dump := flag.String("dump", "", "write the decoded catalog here as JSON")
	flag.Parse()
	raw, err := read(*src)
	if err != nil {
		fail(err)
	}
	catalog, err := decode(raw)
	if err != nil {
		fail(err)
	}
	if *dump != "" {
		out, _ := json.Marshal(catalog)
		if err := os.WriteFile(*dump, out, 0o644); err != nil {
			fail(err)
		}
	}
	report(models(catalog))
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, "modelcatalog:", err)
	os.Exit(1)
}

func read(src string) ([]byte, error) {
	if !strings.HasPrefix(src, "https://") && !strings.HasPrefix(src, "http://") {
		return os.ReadFile(src)
	}
	client := &http.Client{Timeout: time.Minute}
	resp, err := client.Get(src)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("%s: %s", src, resp.Status)
	}
	return io.ReadAll(resp.Body)
}

// models are the chat models of the providers this report reads.
func models(catalog map[string]map[string]map[string]any) []model {
	wanted := map[string]bool{}
	for p := range vendors {
		wanted[p] = true
	}
	for _, p := range makers {
		wanted[p] = true
	}
	out := []model{}
	for provider, entries := range catalog {
		if !wanted[provider] {
			continue
		}
		for id, m := range entries {
			if kind, _ := m["kind"].(string); kind != "" {
				continue // image, embedding, judge, …: not a chat model
			}
			x := model{Provider: provider, ID: id, Window: number(m["contextWindow"])}
			x.Reasons, _ = m["reasoning"].(bool)
			if thinking, ok := m["thinking"].(map[string]any); ok {
				for _, e := range list(thinking["efforts"]) {
					x.Efforts = append(x.Efforts, e)
				}
			}
			if cost, ok := m["cost"].(map[string]any); ok {
				if long, ok := cost["longContext"].(map[string]any); ok {
					x.Threshold = number(long["inputThreshold"])
				}
			}
			out = append(out, x)
		}
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Provider != out[j].Provider {
			return out[i].Provider < out[j].Provider
		}
		return out[i].ID < out[j].ID
	})
	return out
}

func report(all []model) {
	var risk, unknown, window, efforts, price []string
	for _, m := range all {
		for _, vendor := range vendors[m.Provider] {
			got := ai.ContextWindowFor(vendor, m.ID)
			switch {
			case m.Window > 0 && got > m.Window:
				risk = append(risk, fmt.Sprintf("%s\t%s\t2ai %s, served %s", vendor, m.ID, k(got), k(m.Window)))
			case m.Window > 0 && got == 0:
				unknown = append(unknown, fmt.Sprintf("%s\t%s\twindow: catalog %s", vendor, m.ID, k(m.Window)))
			case m.Window > 0 && got*10 < m.Window*9 && vendor != ai.VendorClaudeCode:
				window = append(window, fmt.Sprintf("%s\t%s\t2ai %s, catalog %s", vendor, m.ID, k(got), k(m.Window)))
			}
		}
		if !slices.Contains(makers, m.Provider) {
			continue
		}
		levels, known := ai.ReasoningEffortsFor(m.Provider, m.ID)
		catalog := m.Efforts
		if catalog == nil {
			catalog = []string{}
		}
		switch {
		case !known && (len(catalog) > 0 || m.Reasons):
			unknown = append(unknown, fmt.Sprintf("%s\t%s\tefforts: catalog %s", m.Provider, m.ID, levelsText(catalog, m.Reasons)))
		case known && !slices.Equal(levels, catalog):
			efforts = append(efforts, fmt.Sprintf("%s\t%s\t2ai %s, catalog %s", m.Provider, m.ID, levelsText(levels, true), levelsText(catalog, m.Reasons)))
		}
		if got := ai.LongContextThresholdFor(m.Provider, m.ID); got != m.Threshold && m.Threshold > 0 {
			price = append(price, fmt.Sprintf("%s\t%s\t2ai %s, catalog %s", m.Provider, m.ID, k(got), k(m.Threshold)))
		}
	}
	section("Window larger than the vendor serves (fix: a run summarizes too late and fails)", risk)
	section("Unknown to 2ai (add the family)", unknown)
	section("Effort levels differ from the maker's", efforts)
	section("Long-context price step differs", price)
	section("Window under 90% of the catalog's (check: a vendor table or a smaller serving vendor may explain it)", window)
}

func section(title string, lines []string) {
	fmt.Printf("## %s: %d\n", title, len(lines))
	for _, line := range lines {
		fmt.Println(line)
	}
	fmt.Println()
}

func levelsText(levels []string, reasons bool) string {
	if len(levels) > 0 {
		return strings.Join(levels, ",")
	}
	if reasons {
		return "(reasons, no level)"
	}
	return "(no reasoning)"
}

func k(n int) string {
	if n == 0 {
		return "0"
	}
	if n%1000 == 0 {
		return fmt.Sprintf("%dk", n/1000)
	}
	return fmt.Sprint(n)
}

func number(v any) int {
	f, _ := v.(float64)
	return int(f)
}

func list(v any) []string {
	items, _ := v.([]any)
	out := []string{}
	for _, item := range items {
		if s, ok := item.(string); ok {
			out = append(out, s)
		}
	}
	return out
}

// The packed catalog (oh-my-pi packages/catalog/src/model-catalog-format.ts,
// version 1): interned nodes, keys and boolean shapes. A row's compat delta is
// not read; this report needs no compat field.
type packed struct {
	Version       int               `json:"version"`
	Keys          []string          `json:"keys"`
	BooleanShapes [][]int           `json:"booleanShapes"`
	Nodes         []json.RawMessage `json:"nodes"`
	Providers     []json.RawMessage `json:"providers"`
}

type decoder struct {
	data  packed
	nodes [][]json.RawMessage
}

func decode(raw []byte) (map[string]map[string]map[string]any, error) {
	var d decoder
	if err := json.Unmarshal(raw, &d.data); err != nil {
		return nil, err
	}
	if d.data.Version != 1 {
		return nil, fmt.Errorf("unsupported catalog version %d", d.data.Version)
	}
	d.nodes = make([][]json.RawMessage, len(d.data.Nodes))
	for i, n := range d.data.Nodes {
		if err := json.Unmarshal(n, &d.nodes[i]); err != nil {
			return nil, fmt.Errorf("node %d: %w", i, err)
		}
	}
	out := map[string]map[string]map[string]any{}
	for _, p := range d.data.Providers {
		var row []json.RawMessage
		if err := json.Unmarshal(p, &row); err != nil || len(row) < 3 {
			return nil, fmt.Errorf("provider row: %v", err)
		}
		var name string
		var rows [][]json.RawMessage
		if err := json.Unmarshal(row[0], &name); err != nil {
			return nil, err
		}
		if err := json.Unmarshal(row[2], &rows); err != nil {
			return nil, err
		}
		constants, _ := d.value(row[1]).(map[string]any)
		models := map[string]map[string]any{}
		for _, r := range rows {
			var id string
			var ref []int
			if len(r) < 2 || json.Unmarshal(r[0], &id) != nil || json.Unmarshal(r[1], &ref) != nil || len(ref) != 1 {
				return nil, fmt.Errorf("%s: invalid model row", name)
			}
			m := map[string]any{"id": id}
			for key, v := range constants {
				m[key] = v
			}
			d.object(m, d.nodes[ref[0]])
			models[id] = m
		}
		out[name] = models
	}
	return out, nil
}

// value decodes one encoded value: a scalar, or a one-element array naming a
// node (-1 is an explicit undefined).
func (d *decoder) value(raw json.RawMessage) any {
	var ref []int
	if json.Unmarshal(raw, &ref) == nil && len(ref) == 1 {
		if ref[0] < 0 {
			return nil
		}
		node := d.nodes[ref[0]]
		var kind int
		_ = json.Unmarshal(node[0], &kind)
		if kind == 1 {
			var entries []json.RawMessage
			_ = json.Unmarshal(node[1], &entries)
			out := make([]any, len(entries))
			for i, e := range entries {
				out[i] = d.value(e)
			}
			return out
		}
		out := map[string]any{}
		d.object(out, node)
		return out
	}
	var v any
	_ = json.Unmarshal(raw, &v)
	return v
}

// object fills o from an object node: [0, [key, value, …], shape, bits].
func (d *decoder) object(o map[string]any, node []json.RawMessage) {
	var pairs []json.RawMessage
	_ = json.Unmarshal(node[1], &pairs)
	for i := 0; i+1 < len(pairs); i += 2 {
		var key int
		_ = json.Unmarshal(pairs[i], &key)
		o[d.data.Keys[key]] = d.value(pairs[i+1])
	}
	var shape int
	var bits []uint32
	_ = json.Unmarshal(node[2], &shape)
	_ = json.Unmarshal(node[3], &bits)
	for i, key := range d.data.BooleanShapes[shape] {
		o[d.data.Keys[key]] = bits[i>>5]>>(i%32)&1 == 1
	}
}
