// An offline model run through the same native runtime used by applications.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"

	"github.com/2found/2ai/agentcore"
	"github.com/2found/2ai/ai"
)

func main() {
	provider := &ai.FallbackProvider{Candidates: []ai.FallbackCandidate{{
		Model: json.RawMessage(`{"id":"demo","provider":"scripted"}`),
		Stream: ai.ScriptedStream(ai.Message{
			Role: "assistant", StopReason: "stop",
			Content: ai.BlockContent(ai.ContentBlock{Type: "text", Text: "Hello from 2ai."}),
		}),
	}}}
	agent, err := agentcore.New(agentcore.Config{NativeProvider: provider, Model: "demo"})
	if err != nil {
		log.Fatal(err)
	}
	result, err := agent.Prompt(context.Background(), "Say hello.")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Final)
}
