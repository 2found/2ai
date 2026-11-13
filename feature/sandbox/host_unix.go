//go:build unix

package sandbox

import (
	"os/exec"
	"syscall"
)

// setProcessGroup makes the command the leader of a new process group, so
// everything it spawns — a shell's background jobs included — can be signalled
// as one unit when the timeout fires.
func setProcessGroup(cmd *exec.Cmd) {
	if cmd.SysProcAttr == nil {
		cmd.SysProcAttr = &syscall.SysProcAttr{}
	}
	cmd.SysProcAttr.Setpgid = true
}

// killProcessGroup SIGKILLs the whole group (negative pid). It falls back to
// killing just the process if the group is already gone.
func killProcessGroup(cmd *exec.Cmd) error {
	if cmd.Process == nil {
		return nil
	}
	if err := syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL); err != nil {
		return cmd.Process.Kill()
	}
	return nil
}
