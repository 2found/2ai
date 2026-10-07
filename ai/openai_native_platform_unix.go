//go:build darwin || linux

package ai

import (
	"runtime"

	"golang.org/x/sys/unix"
)

func completionsUserAgent() string {
	var info unix.Utsname
	if unix.Uname(&info) != nil {
		return "pi (browser)"
	}
	arch := runtime.GOARCH
	if arch == "amd64" {
		arch = "x64"
	}
	if arch == "386" {
		arch = "ia32"
	}
	return "pi (" + runtime.GOOS + " " + unix.ByteSliceToString(info.Release[:]) + "; " + arch + ")"
}
