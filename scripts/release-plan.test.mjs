import { test } from "node:test";
import assert from "node:assert/strict";
import { releasePlan } from "./release-plan.mjs";
const sha = "a".repeat(40), old = "b".repeat(40);
const input = {version: "1.2.3", refType: "branch", refName: "main", commit: sha, tags: []};
test("new version releases, unchanged version is a no-op, incomplete release resumes", () => {
  assert.deepEqual(releasePlan(input), {version: "1.2.3", tag: "v1.2.3", commit: sha, release: true, createTag: true});
  const tags = [{tag: "v1.2.3", commit: old}];
  assert.equal(releasePlan({...input, tags}).release, false);
  assert.equal(releasePlan({...input, tags: [{tag: "v1.2.3", commit: sha}]}).createTag, false);
  assert.equal(releasePlan({...input, tags: [{tag: "v1.2.3", commit: sha}], published: true}).release, false);
});
test("manual tag recovery checks identity and cannot downgrade or accept malformed versions", () => {
  const tags = [{tag: "v1.2.3", commit: sha}];
  assert.equal(releasePlan({...input, refType: "tag", refName: "v1.2.3", tags}).release, true);
  assert.throws(() => releasePlan({...input, refType: "tag", refName: "v1.2.4", tags}), /match/);
  assert.throws(() => releasePlan({...input, refType: "tag", refName: "v1.2.3", tags: [{tag: "v1.2.3", commit: old}]}), /commit/);
  assert.throws(() => releasePlan({...input, tags: [{tag: "v1.2.4", commit: old}]}), /older/);
  assert.throws(() => releasePlan({...input, refName: "feature"}), /main/);
  for (const version of ["1.2", "01.2.3", "1.2.3-beta.1", "1.2.3\n", "bad"])
    assert.throws(() => releasePlan({...input, version}), /semver/);
});
