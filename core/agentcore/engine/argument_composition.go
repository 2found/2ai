package engine

import (
	"net/url"
	"strconv"
	"sync"

	"github.com/santhosh-tekuri/jsonschema/v6"
)

// Pi merges a compound keyword's evaluated properties/items only when the
// entire keyword succeeds. The backend otherwise merges each successful
// allOf branch, and the first successful oneOf branch even if another matches.
// A containing validation frame makes that merge conditional on the group.
// Keep original child schemas/locations so references and diagnostics retain
// their source paths. The empty compiled base supplies valid resource metadata
// for recursive/dynamic references crossing the extra frame.
var argumentCompositionBase = sync.OnceValues(func() (*jsonschema.Schema, error) {
	// Keep this private resource out of the caller's reference registry.
	compiler := jsonschema.NewCompiler()
	compiler.DefaultDraft(jsonschema.Draft2019)
	const empty = "https://agentcore.local/pi-composition-frame.json"
	if err := compiler.AddResource(empty, map[string]any{}); err != nil {
		return nil, err
	}
	return compiler.Compile(empty)
})

func registerArgumentComposition(compiler *jsonschema.Compiler, locations map[string]string) error {
	base, err := argumentCompositionBase()
	if err != nil {
		return err
	}
	compiler.RegisterVocabulary(&jsonschema.Vocabulary{
		URL: "https://agentcore.local/pi-composition-semantics",
		Compile: func(ctx *jsonschema.CompilerContext, obj map[string]any) (jsonschema.SchemaExt, error) {
			if _, all := obj["allOf"]; !all {
				if _, one := obj["oneOf"]; !one {
					return nil, nil
				}
			}
			current := ctx.Enqueue(nil)
			location, err := url.Parse(current.Location)
			if err != nil {
				return nil, err
			}
			count := len(current.AllOf)
			// compilerSchema appends isolated type/const/enum checks. They are sibling
			// keywords, not branches in the caller's allOf, so leave those outside.
			for count > 0 {
				original, synthetic := locations[location.Fragment+"/allOf/"+strconv.Itoa(count-1)]
				if !synthetic || original != location.Fragment {
					break
				}
				count--
			}
			if count > 0 {
				group := *base
				group.Bool, group.Location = nil, current.Location
				group.AllOf = append([]*jsonschema.Schema(nil), current.AllOf[:count]...)
				current.AllOf = append([]*jsonschema.Schema{&group}, current.AllOf[count:]...)
			}
			if len(current.OneOf) > 0 {
				group := *base
				group.Bool, group.Location = nil, current.Location
				group.OneOf = current.OneOf
				current.OneOf = nil
				current.AllOf = append(current.AllOf, &group)
			}
			return nil, nil
		},
	})
	return nil
}
