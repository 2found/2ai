// Package agentcore is a reusable, product-agnostic agent runtime: the turn
// loop, tool calling, hooks, permissions, memory, and the Agent Definition. It
// knows nothing about analytics or any other product. A consumer injects a
// ToolSet, a Policy, a MemoryStore, and an AgentDefinition; agentcore drives
// them.
//
// # Boundaries
//
// Two rules define this package, and both are tests rather than promises (see
// boundary_test.go):
//
//   - Composition depends only on shared runtime modules. Provider-neutral
//     contracts live in ai/protocol and are re-exported here during migration;
//     ai no longer imports agentcore. The native loop lives in engine and
//     reusable native compaction/checkpoint policy lives in host.
//   - The kernel names no plugin. Delete every package under agentcore/plugins
//     and this package still compiles, still runs, and still passes its tests —
//     it just does less.
//
// # Composition
//
// An agent is not constructed, it is composed. A [Plugin] contributes to a
// [Registry]; [Build] turns a plugin set into an [Agent]. There is no privileged
// core to patch — agentcore's own behavior is just the default plugin set, and
// any of it can be replaced by registering something else in its place.
//
//	agent, err := agentcore.Build(
//		agentcore.ModelPlugin{...},      // seam: provider, ladder, retry
//		agentcore.DefinitionPlugin{...}, // seam: persona, skills, limits
//		agentcore.PolicyPlugin{...},     // seam: the permission gate
//		todo.Plugin{...},                // extension: adds a tool and a step interceptor
//	)
//
// [New] is the same composition reached through a flat [Config] instead of a
// plugin list; both write through the same exported Registry setters, and
// agentcore/plugins/preset carries a parity test proving they agree.
//
// A plugin contributes in exactly one of three ways, and which one it is can be
// read off its Register body:
//
//   - Seam — r.SetX(...). Keyed, exactly one provider, a second claim is a build
//     error naming both plugins. Bought property: replaceability.
//   - Extension — r.AddExtension(...). Ordered by explicit Priority, never by
//     registration order. This is how a capability reaches a running agent.
//   - Additive — r.AddTools / r.AddHooks. Accumulates.
//
// # Reading this package
//
// The engine subpackage is the native Go port of Pi's state machine, using the
// lossless ai transcript. It is replacing this legacy flat kernel in stages;
// it does not call the TypeScript worker. Its dependencies are checked separately.
//
// The runtime root is deliberately flat: these files are not independent
// concerns that happen to sit together, they are one machine reached through
// unexported fields of one [Agent]. Black-box composition tests live in
// agentcore/integration; the real production boundary is agentcore/plugins, so the
// organizing question here is not "which folder?" but "why is this file in the
// kernel and not a plugin?". Every file answers with one of four words — loop,
// contract, log, or seam default — and README.md holds the map, which
// TestEveryKernelFileJustifiesItself gates.
//
// The layers that map runs through, outermost first:
//
//   - Composition — doc.go, plugin.go, compose.go, agent.go, seams.go.
//   - Contracts — provider.go, provider_session.go, tool.go, permission.go,
//     definition.go, hooks.go, extension.go, memory.go, env.go. The types a
//     plugin or provider outside this repo implements.
//   - The loop — loop.go, turn.go, tooldispatch.go, result.go, prompt.go,
//     skill_tool.go.
//   - Durable state — session.go, session_tree.go, memsession.go,
//     compaction.go, fork.go, lab.go. The loop is the only writer; everything
//     else reduces the log.
//
// One file sits outside those layers on purpose: faux.go, the providers that
// make the loop, the hooks and the gate exercisable with no network and no key
// — [FauxProvider] scripted, [ReplayProvider] over a recorded transcript. They
// ship in the kernel rather than a test package because the plugins are tested
// against them too, and a plugin proving its behavior against the real loop is
// the point of the whole split.
package agentcore
