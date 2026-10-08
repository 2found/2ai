# Releases

`VERSION` owns the stable SemVer number. A push to `main` with an untagged version
runs the race tests, vet and scripted example, then creates the immutable
`v<VERSION>` tag and publishes a GitHub Release. Pushes that retain a previously
tagged version do not release. CI never guesses a version from commit messages.

2ai is a Go library: the tag distributes the Go module; there is no application
binary or desktop installer. A release also contains a versioned docs archive,
`release.json`, `DOWNLOADS.md` and `checksums.txt`. The release body is the download
page and links documentation to the same tag, rather than the moving main branch.

Use an explicit published tag in consuming modules:

```sh
go get github.com/2found/2ai@v<version>
```

The first configured version is `0.1.0`; it is not published merely by adding the
workflow. Keep consumers pinned until their own compatibility checks pass.

Tags are created only after verification. Distribution is a dependent job in the
same workflow because a tag pushed with `GITHUB_TOKEN` does not start another
workflow. Manual `v*` tag pushes use the same path and must match `VERSION`.
To resume a failed distribution, rerun failed jobs or dispatch `release.yml`
with the existing tag. The planner resumes an unpublished tag on the same commit;
it never moves tags, rebuilds published releases or promotes an older version.

GitHub Actions needs `contents: write` for tags and releases. No publish credential
or provider API key is needed. Releases do not upgrade dependent products or
deploy consuming applications.

The 2found website consumes published release catalogs hourly. Optional secret
`WEBSITE_DISPATCH_TOKEN`, scoped only to dispatch `2found/2found.dev`, triggers an
immediate refresh. A notification failure does not undo the product release;
the scheduled consumer repairs missed refreshes. Never store that token in source.
