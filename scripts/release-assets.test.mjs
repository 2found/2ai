import { test } from "node:test";
import assert from "node:assert/strict";
import { downloadManifest } from "./release-assets.mjs";
test("download manifest binds checksums and URLs to the exact version and source", () => {
  const result = downloadManifest("2found/2agent", "0.1.0", "a".repeat(40), [{name: "soot_0.1.0_linux_amd64.tar.gz", bytes: Buffer.from("fixture")}]);
  assert.equal(result.artifacts[0].url, "https://github.com/2found/2agent/releases/download/v0.1.0/soot_0.1.0_linux_amd64.tar.gz");
  assert.equal(result.artifacts[0].sha256.length, 64);
  assert.equal(result.artifacts[0].size, 7);
  assert.equal(result.docs, "https://github.com/2found/2agent/tree/v0.1.0");
  for (const name of ["../private.env", "bad name", "file\n"])
    assert.throws(() => downloadManifest("2found/2agent", "0.1.0", "a".repeat(40), [{name, bytes: Buffer.from("x")}]), /artifact/);
  assert.throws(() => downloadManifest("2found/other", "0.1.0", "a".repeat(40), []), /identity/);
  assert.throws(() => downloadManifest("2found/2ai", "0.1.0", "bad", []), /identity/);
});
