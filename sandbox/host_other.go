//go:build !unix

package sandbox

import "os/exec"

// On a platform without POSIX process groups the timeout falls back to os/exec's
// default: the direct child is killed, anything it detached is not.
func setProcessGroup(*exec.Cmd) {}

func killProcessGroup(cmd *exec.Cmd) error {
	if cmd.Process == nil {
		return nil
	}
	return cmd.Process.Kill()
}
