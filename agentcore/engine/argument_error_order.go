package engine

import (
	"bytes"
	"encoding/json"
	"net/url"
	"slices"
	"strconv"
	"strings"

	"github.com/2found/2ai/jsonjs"
	"github.com/santhosh-tekuri/jsonschema/v6"
	"github.com/santhosh-tekuri/jsonschema/v6/kind"
)

// Pi traverses schema keywords in engine order, properties in JavaScript key
// order, and stops collecting diagnostics after eight errors.
var validationKeywords = strings.Fields("type required additionalProperties dependencies dependentRequired dependentSchemas patternProperties properties propertyNames minProperties maxProperties additionalItems contains items maxContains maxItems minContains minItems prefixItems uniqueItems maxLength minLength format pattern exclusiveMaximum exclusiveMinimum maximum minimum multipleOf $ref $recursiveRef $dynamicRef const enum if not allOf anyOf oneOf unevaluatedItems unevaluatedProperties")

func validationKeywordRank(keyword string) int {
	if i := slices.Index(validationKeywords, keyword); i >= 0 {
		return i
	}
	return len(validationKeywords)
}

func validationObject(raw json.RawMessage) ([]string, map[string]json.RawMessage) {
	keys := []string{}
	values := map[string]json.RawMessage{}
	properties, err := jsonjs.DecodeObjectProperties(raw)
	if err != nil {
		return keys, values
	}
	for _, property := range properties {
		if _, exists := values[property.Name]; !exists {
			keys = append(keys, property.Name)
		}
		values[property.Name] = property.Value
	}
	return keys, values
}

type validationDiagnostic struct {
	err     *jsonschema.ValidationError
	order   []int
	message string
	names   []string
}

func orderedValidationLines(err *jsonschema.ValidationError, schema, arguments json.RawMessage) []string {
	diagnostics := []validationDiagnostic{}
	type scope struct {
		schema, arguments json.RawMessage
		base              string
		depth             int
		prefix            []int
	}
	var visit func(*jsonschema.ValidationError, scope)
	visit = func(e *jsonschema.ValidationError, current scope) {
		switch e.ErrorKind.(type) {
		case *kind.Schema, *kind.Group, *kind.AllOf:
			for _, child := range e.Causes {
				visit(child, current)
			}
			return
		}
		position := e
		if _, ok := e.ErrorKind.(*kind.PropertyNames); ok {
			copy := *e
			location, _ := url.Parse(e.SchemaURL)
			location.Fragment = strings.TrimSuffix(location.Fragment, "/propertyNames")
			copy.SchemaURL = location.String()
			position = &copy
		}
		order, node, instance, parents := validationOrder(position, current.schema, current.arguments, current.base, current.depth)
		for _, parent := range parents {
			parent.order = append(append([]int(nil), current.prefix...), parent.order...)
			exists := slices.ContainsFunc(diagnostics, func(d validationDiagnostic) bool { return slices.Equal(d.order, parent.order) })
			if !exists {
				diagnostics = append(diagnostics, parent)
			}
		}
		if order == nil {
			return
		} // Pi hides diagnostics from then and unevaluated child checks.
		order = append(append([]int(nil), current.prefix...), order...)
		if names, ok := e.ErrorKind.(*kind.PropertyNames); ok {
			path := append(append([]string(nil), e.InstanceLocation...), names.Property)
			for _, child := range e.Causes {
				visit(validationRebaseError(child, path), current)
			}
			order = append(order, int(^uint(0)>>1))
			index := slices.IndexFunc(diagnostics, func(d validationDiagnostic) bool { return slices.Equal(d.order, order) })
			if index < 0 {
				index = len(diagnostics)
				diagnostics = append(diagnostics, validationDiagnostic{err: position, order: order})
			}
			diagnostic := &diagnostics[index]
			diagnostic.names = append(diagnostic.names, names.Property)
			keys, _ := validationObject(instance)
			slices.SortFunc(diagnostic.names, func(a, b string) int { return slices.Index(keys, a) - slices.Index(keys, b) })
			diagnostic.message = "property names " + strings.Join(diagnostic.names, ", ") + " are invalid"
			return
		}
		if reference, ok := e.ErrorKind.(*kind.Reference); ok {
			target := validationSchemaAt(schema, validationPointer(reference.URL))
			for _, child := range e.Causes {
				visit(child, scope{schema: target, arguments: instance, base: reference.URL, depth: len(e.InstanceLocation), prefix: order})
			}
			return
		}
		if minimum, ok := e.ErrorKind.(*kind.MinContains); ok && len(minimum.Got) == 0 {
			// Pi checks contains separately before the explicit minContains bound.
			// Both failures are observable, with other array keywords between them.
			containsOrder := append([]int(nil), order...)
			containsOrder[len(containsOrder)-1] = validationKeywordRank("contains")
			diagnostics = append(diagnostics, validationDiagnostic{err: e, order: containsOrder})
		}
		switch e.ErrorKind.(type) {
		case *kind.AnyOf, *kind.OneOf:
			for _, child := range e.Causes {
				visit(child, current)
			}
			order = append(order, int(^uint(0)>>1))
		}

		diagnostic := validationDiagnostic{err: e, order: order}
		if _, ok := e.ErrorKind.(*kind.Type); ok {
			_, fields := validationObject(node)
			var types []string
			if json.Unmarshal(fields["type"], &types) == nil && types != nil {
				// The validator stores types as a set. Pi's wording depends on
				// the original array shape and insertion order, even for one type.
				diagnostic.message = "must be either " + strings.Join(types, " or ")
			}
		}
		if keyword, property, repeats := validationDependency(e); keyword != "" {
			_, fields := validationObject(node)
			_, dependencies := validationObject(fields[keyword])
			required := []string{}
			if value, err := jsonjs.DecodeJSON(dependencies[property]); err == nil {
				if names, ok := value.([]any); ok {
					for _, name := range names {
						if name, ok := name.(string); ok {
							required = append(required, name)
						}
					}
				}
			}
			diagnostic.message = "must have properties " + strings.Join(required, ", ") + " when property " + property + " is present"
			// Legacy dependencies stops after the first missing key; the newer
			// dependentRequired keyword emits one identical error per missing key.
			for i := 1; i < repeats; i++ {
				diagnostics = append(diagnostics, diagnostic)
			}
		}
		diagnostics = append(diagnostics, diagnostic)
	}
	visit(err, scope{schema: schema, arguments: arguments})
	slices.SortStableFunc(diagnostics, func(a, b validationDiagnostic) int { return slices.Compare(a.order, b.order) })
	lines := []string{}
	for _, diagnostic := range diagnostics {
		if diagnostic.message != "" {
			lines = append(lines, validationLine(validationPath(diagnostic.err.InstanceLocation), diagnostic.message))
		} else {
			lines = append(lines, validationLines(diagnostic.err)...)
		}
		if len(lines) >= 8 {
			return lines[:8]
		}
	}
	return lines
}

// SchemaURL names the containing schema, while InstanceLocation supplies array
// indices and additional-property names absent from that schema pointer.
func validationOrder(e *jsonschema.ValidationError, schema, instance json.RawMessage, base string, depth int) ([]int, json.RawMessage, json.RawMessage, []validationDiagnostic) {
	parents := []validationDiagnostic{}
	parts := validationPointer(e.SchemaURL)
	baseParts := validationPointer(base)
	if len(parts) >= len(baseParts) && slices.Equal(parts[:len(baseParts)], baseParts) {
		parts = parts[len(baseParts):]
	}
	order := []int{}
	for len(parts) > 0 {
		keyword := parts[0]
		parts = parts[1:]
		rankKeyword := keyword
		if keyword == "then" || keyword == "else" {
			rankKeyword = "if"
		}
		order = append(order, validationKeywordRank(rankKeyword))
		if keyword == "unevaluatedItems" || keyword == "unevaluatedProperties" {
			// Pi checks these children in private contexts, then reports only
			// one summary at the enclosing array/object, regardless of count.
			message := "must not have unevaluated items"
			if keyword == "unevaluatedProperties" {
				message = "must not have unevaluated properties"
			}
			parent := &jsonschema.ValidationError{InstanceLocation: append([]string(nil), e.InstanceLocation[:depth]...)}
			parents = append(parents, validationDiagnostic{err: parent, order: append([]int(nil), order...), message: message})
			return nil, nil, nil, parents
		}
		if keyword == "then" || keyword == "else" {
			parent := &jsonschema.ValidationError{InstanceLocation: append([]string(nil), e.InstanceLocation[:depth]...)}
			parentOrder := append(append([]int(nil), order...), int(^uint(0)>>1))
			parents = append(parents, validationDiagnostic{err: parent, order: parentOrder, message: "must match \"" + keyword + "\" schema"})
			// Pi collects then-branch errors in a private context and emits
			// only the summary; else-branch errors use the parent's context.
			if keyword == "then" {
				return nil, nil, nil, parents
			}
			order = append(order, 0)
		}
		_, fields := validationObject(schema)
		schema = fields[keyword]
		switch keyword {
		case "dependencies", "dependentSchemas":
			if len(parts) == 0 {
				break
			}
			key := parts[0]
			parts = parts[1:]
			keys, children := validationObject(schema)
			order = append(order, slices.Index(keys, key))
			schema = children[key]
		case "properties", "patternProperties":
			if len(parts) == 0 {
				break
			}
			key := parts[0]
			parts = parts[1:]
			keys, children := validationObject(schema)
			order = append(order, slices.Index(keys, key))
			schema = children[key]
			if depth < len(e.InstanceLocation) {
				keys, values := validationObject(instance)
				if keyword == "patternProperties" {
					order = append(order, slices.Index(keys, e.InstanceLocation[depth]))
				}
				instance = values[e.InstanceLocation[depth]]
				depth++
			}
		case "allOf", "anyOf", "oneOf", "items", "prefixItems", "additionalItems":
			var array []json.RawMessage
			if json.Unmarshal(schema, &array) == nil && len(parts) > 0 {
				index, _ := strconv.Atoi(parts[0])
				parts = parts[1:]
				order = append(order, index)
				if index >= 0 && index < len(array) {
					schema = array[index]
				}
			}
			if keyword == "items" || keyword == "prefixItems" || keyword == "additionalItems" {
				if depth < len(e.InstanceLocation) {
					index, _ := strconv.Atoi(e.InstanceLocation[depth])
					depth++
					order = append(order, index)
					var values []json.RawMessage
					if json.Unmarshal(instance, &values) == nil && index >= 0 && index < len(values) {
						instance = values[index]
					}
				}
			}
		case "propertyNames":
			if depth < len(e.InstanceLocation) {
				keys, _ := validationObject(instance)
				name := e.InstanceLocation[depth]
				depth++
				order = append(order, slices.Index(keys, name))
				instance = jsonjs.QuoteString(name)
			}
		case "additionalProperties":
			// The Go validator reports schema-valued additional-property failures
			// only at the child. Pi also emits one parent summary after all children.
			if depth < len(e.InstanceLocation) {
				parent := &jsonschema.ValidationError{
					InstanceLocation: append([]string(nil), e.InstanceLocation[:depth]...),
					ErrorKind:        &kind.AdditionalProperties{},
				}
				parentOrder := append(append([]int(nil), order...), int(^uint(0)>>1))
				parents = append(parents, validationDiagnostic{err: parent, order: parentOrder})
				keys, values := validationObject(instance)
				key := e.InstanceLocation[depth]
				depth++
				order = append(order, slices.Index(keys, key))
				instance = values[key]
			}
		}
	}
	keyword := ""
	if path := e.ErrorKind.KeywordPath(); len(path) > 0 {
		keyword = path[0]
	}
	if _, ok := e.ErrorKind.(*kind.Not); ok {
		keyword = "not"
	}
	if dependencyKeyword, property, _ := validationDependency(e); dependencyKeyword != "" {
		_, fields := validationObject(schema)
		keys, _ := validationObject(fields[dependencyKeyword])
		order = append(order, validationKeywordRank(dependencyKeyword), slices.Index(keys, property))
	} else {
		order = append(order, validationKeywordRank(keyword))
	}
	return order, schema, instance, parents
}

// References retain their caller's ordering prefix while traversing the target
// schema relative to its own pointer. Decode pointer tokens only after splitting.
func validationPointer(location string) []string {
	parsed, err := url.Parse(location)
	if err != nil || parsed.Fragment == "" {
		return nil
	}
	parts := strings.Split(strings.TrimPrefix(parsed.Fragment, "/"), "/")
	for i := range parts {
		parts[i] = strings.ReplaceAll(strings.ReplaceAll(parts[i], "~1", "/"), "~0", "~")
	}
	return parts
}

func validationSchemaAt(schema json.RawMessage, parts []string) json.RawMessage {
	for _, part := range parts {
		if bytes.HasPrefix(bytes.TrimSpace(schema), []byte("[")) {
			var items []json.RawMessage
			index, err := strconv.Atoi(part)
			if err != nil || json.Unmarshal(schema, &items) != nil || index < 0 || index >= len(items) {
				return nil
			}
			schema = items[index]
		} else {
			_, fields := validationObject(schema)
			schema = fields[part]
		}
	}
	return schema
}

// The validator checks property names as standalone strings. Restore their
// enclosing object path without mutating the validator's original error tree.
func validationRebaseError(err *jsonschema.ValidationError, prefix []string) *jsonschema.ValidationError {
	result := *err
	result.InstanceLocation = append(append([]string(nil), prefix...), err.InstanceLocation...)
	result.Causes = make([]*jsonschema.ValidationError, len(err.Causes))
	for i, child := range err.Causes {
		result.Causes[i] = validationRebaseError(child, prefix)
	}
	return &result
}

func validationDependency(err *jsonschema.ValidationError) (keyword, property string, repeats int) {
	switch dependency := err.ErrorKind.(type) {
	case *kind.Dependency:
		return "dependencies", dependency.Prop, 1
	case *kind.DependentRequired:
		return "dependentRequired", dependency.Prop, len(dependency.Missing)
	default:
		return "", "", 0
	}
}
