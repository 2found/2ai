import { execFileSync } from "node:child_process";
import { appendFileSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

const stable = /^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$/;
export function compareVersions(left, right) {
  if (typeof left !== "string" || typeof right !== "string" || left !== left.trim() || right !== right.trim() || !stable.test(left) || !stable.test(right)) throw new Error("Expected stable semver");
  const a = left.split(".").map(BigInt), b = right.split(".").map(BigInt);
  for (let i = 0; i < 3; i++) if (a[i] !== b[i]) return a[i] > b[i] ? 1 : -1;
  return 0;
}

// Pure policy: an ordinary main push never invents a version. A retry can
// resume an unpublished tag on the same source commit without moving the tag.
export function releasePlan({version, refType, refName, commit, tags, published = false}) {
  compareVersions(version, version);
  const tag = `v${version}`;
  if (refType === "tag" && refName !== tag) throw new Error("Tag must match source version");
  if (refType !== "tag" && (refType !== "branch" || refName !== "main")) throw new Error("Release requires main or a version tag");
  const existing = tags.find(t => t.tag === tag);
  const newer = tags.some(t => stable.test(t.tag.slice(1)) && compareVersions(t.tag.slice(1), version) > 0);
  if (newer) throw new Error("Refusing a version older than an existing release tag");
  if (refType === "tag" && (!existing || existing.commit !== commit)) throw new Error("Tag does not resolve to the checked-out commit");
  if (refType === "branch" && existing && existing.commit !== commit)
    return {version, tag, commit, release: false, createTag: false};
  return {version, tag, commit, release: !published, createTag: !existing && !published};
}

export function sourceVersion() {
  return readFileSync("VERSION", "utf8").trim();
}

function main() {
  if (!process.env.GITHUB_OUTPUT) throw new Error("CI only; use the tests for a local dry run");
  const git = (...args) => execFileSync("git", args, {encoding: "utf8"}).trim();
  const commit = git("rev-parse", "HEAD");
  const remote = git("ls-remote", "--tags", "origin"); // network/auth errors must fail, never mint a replacement tag
  const refs = new Map(remote.split("\n").filter(Boolean).map(line => line.split(/\s+/).reverse()));
  const tags = [...refs].filter(([ref]) => /^refs\/tags\/v/.test(ref) && !ref.endsWith("^{}"))
    .map(([ref, sha]) => ({tag: ref.slice("refs/tags/".length), commit: refs.get(`${ref}^{}`) ?? sha}));
  const version = sourceVersion();
  const input = {version, refType: process.env.RELEASE_REF_TYPE, refName: process.env.RELEASE_REF_NAME, commit, tags};
  let plan = releasePlan(input);
  if (plan.release && !plan.createTag) {
    try {
      const release = JSON.parse(execFileSync("gh", ["api", `repos/${process.env.GITHUB_REPOSITORY}/releases/tags/${plan.tag}`], {encoding: "utf8", stdio: ["ignore", "pipe", "pipe"]}));
      plan = releasePlan({...input, published: release.draft === false});
    } catch (error) {
      if (!String(error.stderr).includes("HTTP 404")) throw new Error("Cannot inspect existing release; retry after fixing GitHub access");
    }
  }
  for (const [key, value] of Object.entries(plan)) appendFileSync(process.env.GITHUB_OUTPUT, `${key}=${value}\n`);
  console.log(`${plan.tag}: ${plan.release ? "release pending" : "already released / no version change"}`);
}
if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  try { main(); } catch (error) { console.error(error.message); process.exitCode = 1; }
}
