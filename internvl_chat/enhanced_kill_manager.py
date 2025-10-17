#!/usr/bin/env python3
import subprocess
import json
import argparse
import time
from typing import List, Dict, Optional


class EnhancedKillManager:
    def __init__(self, ssh_user="eric"):
        self.ssh_user = ssh_user

    def run_ssh_command(self, vm_ip: str, command: str, debug=False) -> tuple:
        """Run SSH command and return (success, output, error)"""
        ssh_target = f"{self.ssh_user}@{vm_ip}"
        ssh_cmd = ["ssh", ssh_target, command]

        if debug:
            print(f"  Running: {command}")

        try:
            result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=30)
            return result.returncode == 0, result.stdout, result.stderr
        except Exception as e:
            return False, "", str(e)

    def find_training_processes(self, vm_ip: str) -> List[Dict]:
        """Find all training-related processes on a VM"""
        print(f"Scanning for training processes on {vm_ip}...")

        # Look for processes that might be training jobs
        patterns = [
            "internvl3_train",
            "python.*train",
            "torchrun",
            "accelerate",
            "deepspeed",
        ]

        processes = []
        for pattern in patterns:
            cmd = f"ps aux | grep -E '{pattern}' | grep -v grep || true"
            success, stdout, _ = self.run_ssh_command(vm_ip, cmd)
            if success and stdout.strip():
                for line in stdout.strip().split("\n"):
                    if line.strip():
                        parts = line.split()
                        if len(parts) >= 11:
                            processes.append(
                                {
                                    "pid": parts[1],
                                    "user": parts[0],
                                    "cpu": parts[2],
                                    "mem": parts[3],
                                    "command": " ".join(parts[10:])[:100] + "..."
                                    if len(" ".join(parts[10:])) > 100
                                    else " ".join(parts[10:]),
                                }
                            )

        return processes

    def find_tmux_sessions(self, vm_ip: str) -> List[Dict]:
        """Find all tmux sessions on a VM"""
        success, stdout, _ = self.run_ssh_command(
            vm_ip, "tmux list-sessions 2>/dev/null || echo 'No sessions'"
        )

        sessions = []
        if success and "No sessions" not in stdout:
            for line in stdout.strip().split("\n"):
                if ":" in line:
                    parts = line.split(":")
                    session_name = parts[0].strip()
                    session_info = ":".join(parts[1:]).strip()
                    sessions.append(
                        {
                            "name": session_name,
                            "info": session_info,
                            "is_training": "training_job" in session_name,
                        }
                    )

        return sessions

    def kill_tmux_session(self, vm_ip: str, session_name: str, force=False) -> bool:
        """Kill a specific tmux session"""
        if force:
            # First try to capture any final output
            self.run_ssh_command(
                vm_ip,
                f"tmux capture-pane -t {session_name} -p > /tmp/{session_name}_final.log 2>/dev/null || true",
            )

        success, _, stderr = self.run_ssh_command(
            vm_ip, f"tmux kill-session -t {session_name} 2>/dev/null"
        )
        return success

    def kill_process_by_pid(self, vm_ip: str, pid: str, signal="TERM") -> bool:
        """Kill a process by PID with specified signal"""
        cmd = f"kill -{signal} {pid} 2>/dev/null"
        success, _, _ = self.run_ssh_command(vm_ip, cmd)
        return success

    def graceful_kill_training(self, vm_ip: str, session_name: str) -> bool:
        """Gracefully kill a training job"""
        print(f"  Attempting graceful shutdown of {session_name}...")

        # Step 1: Send Ctrl+C to the tmux session
        success, _, _ = self.run_ssh_command(
            vm_ip, f"tmux send-keys -t {session_name} C-c"
        )
        if not success:
            print(f"    ✗ Failed to send interrupt signal")
            return False

        # Step 2: Wait a few seconds for graceful shutdown
        print(f"    Waiting 5 seconds for graceful shutdown...")
        time.sleep(1)

        # Step 3: Check if session still exists
        sessions = self.find_tmux_sessions(vm_ip)
        session_exists = any(s["name"] == session_name for s in sessions)

        if not session_exists:
            print(f"    ✓ Session terminated gracefully")
            return True

        # Step 4: If still running, kill the session
        print(f"    Session still running, forcing termination...")
        return self.kill_tmux_session(vm_ip, session_name, force=True)

    def kill_from_job_results(self, results_file="job_results.json", graceful=True):
        """Kill jobs based on saved job results"""
        try:
            with open(results_file, "r") as f:
                results = json.load(f)
        except FileNotFoundError:
            print(f"❌ No job results found in {results_file}")
            print("   Jobs may have been launched manually or results file was deleted")
            return False

        print(f"Killing jobs from {results_file}...")
        print("=" * 50)

        success_count = 0
        total_count = len(results)

        for result in results:
            vm_ip = result.get("vm_ip")
            session_name = result.get("session_name")
            node_rank = result.get("node_rank", "unknown")

            if not vm_ip or not session_name:
                print(f"❌ Invalid result entry: {result}")
                continue

            print(f"\nKilling job on {vm_ip} (rank {node_rank}):")
            print(f"  Session: {session_name}")

            if graceful:
                success = self.graceful_kill_training(vm_ip, session_name)
            else:
                success = self.kill_tmux_session(vm_ip, session_name, force=True)

            if success:
                print(f"  ✓ Successfully killed job on {vm_ip}")
                success_count += 1
            else:
                print(f"  ✗ Failed to kill job on {vm_ip}")

            # Clean up wrapper script if it exists
            wrapper_script = result.get("wrapper_script")
            if wrapper_script:
                self.run_ssh_command(vm_ip, f"rm -f {wrapper_script} 2>/dev/null")

        print(f"\n" + "=" * 50)
        print(f"Kill summary: {success_count}/{total_count} jobs terminated")

        return success_count == total_count

    def kill_all_training_sessions(self, vms: List[str]):
        """Kill all training-related tmux sessions on specified VMs"""
        print("Scanning for training sessions across all VMs...")
        print("=" * 50)

        total_killed = 0

        for vm_ip in vms:
            print(f"\nVM: {vm_ip}")
            sessions = self.find_tmux_sessions(vm_ip)
            training_sessions = [s for s in sessions if s["is_training"]]

            if not training_sessions:
                print("  No training sessions found")
                continue

            print(f"  Found {len(training_sessions)} training session(s):")
            for session in training_sessions:
                print(f"    - {session['name']}: {session['info']}")

            for session in training_sessions:
                success = self.graceful_kill_training(vm_ip, session["name"])
                if success:
                    total_killed += 1

        print(f"\n" + "=" * 50)
        print(f"Total training sessions killed: {total_killed}")

    def kill_all_processes(self, vms: List[str], dry_run=False):
        """Kill all training processes (more aggressive)"""
        print("Scanning for training processes across all VMs...")
        if dry_run:
            print("(DRY RUN - no processes will be killed)")
        print("=" * 50)

        total_killed = 0

        for vm_ip in vms:
            print(f"\nVM: {vm_ip}")
            processes = self.find_training_processes(vm_ip)

            if not processes:
                print("  No training processes found")
                continue

            print(f"  Found {len(processes)} training process(es):")
            for proc in processes:
                print(f"    PID {proc['pid']}: {proc['command']}")

            if not dry_run:
                for proc in processes:
                    success = self.kill_process_by_pid(vm_ip, proc["pid"])
                    if success:
                        print(f"    ✓ Killed PID {proc['pid']}")
                        total_killed += 1
                    else:
                        print(f"    ✗ Failed to kill PID {proc['pid']}")

        if not dry_run:
            print(f"\n" + "=" * 50)
            print(f"Total processes killed: {total_killed}")

    def show_status(self, vms: List[str]):
        """Show current status of training jobs"""
        print("Current training job status:")
        print("=" * 60)

        for vm_ip in vms:
            print(f"\n📍 VM: {vm_ip}")

            # Show tmux sessions
            sessions = self.find_tmux_sessions(vm_ip)
            if sessions:
                print("  Tmux sessions:")
                for session in sessions:
                    status = "🔥 TRAINING" if session["is_training"] else "📋 Other"
                    print(f"    {status} {session['name']}: {session['info']}")
            else:
                print("  No tmux sessions")

            # Show processes
            processes = self.find_training_processes(vm_ip)
            if processes:
                print("  Training processes:")
                for proc in processes:
                    print(
                        f"    🔥 PID {proc['pid']} ({proc['cpu']}% CPU): {proc['command']}"
                    )
            else:
                print("  No training processes found")


def main():
    all_vms = [
        "89.169.108.234",  # 0
        "192.168.0.0",  # 1
        "192.168.0.107",  # 2
        "192.168.0.187",  # 3
        "192.168.0.70",  # 4
        "192.168.0.113",  # 5
        "192.168.0.211",  # 6
        "192.168.0.137",  # 7
        "192.168.0.110",  # 8
        "192.168.0.66",  # 9
        "192.168.0.136",  # 10
        "192.168.0.170",  # 11
        "192.168.0.229",  # 12
        "192.168.0.228",  # 13
        "192.168.0.48",  # 14
        "192.168.0.6",  # 15
        "192.168.0.93",  # 16
        "192.168.0.68",  # 17
        "192.168.0.69",  # 18
        "192.168.0.129",  # 19
        "192.168.0.212",  # 20
        "192.168.0.100",  # 21
        "192.168.0.202",  # 22
        "192.168.0.50",  # 23
        "192.168.0.33",  # 24
        "192.168.0.146",  # 25
        "192.168.0.72",  # 26
        "192.168.0.71",  # 27
    ]

    parser = argparse.ArgumentParser(description="Enhanced Training Job Kill Manager")
    parser.add_argument(
        "--action",
        choices=["kill", "kill-all", "kill-processes", "status"],
        default="kill",
        help="Action to perform",
    )
    parser.add_argument(
        "--vms",
        nargs="+",
        default=all_vms,
        help="VM IPs to operate on",
    )
    parser.add_argument(
        "--graceful",
        action="store_true",
        default=True,
        help="Use graceful shutdown (default)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Force immediate termination"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be killed without actually killing",
    )
    parser.add_argument(
        "--results-file",
        default="job_results.json",
        help="Job results file to use for kill action",
    )

    args = parser.parse_args()

    manager = EnhancedKillManager()

    if args.action == "kill":
        # Kill jobs from saved results
        manager.kill_from_job_results(args.results_file, graceful=not args.force)

    elif args.action == "kill-all":
        # Kill all training sessions
        manager.kill_all_training_sessions(args.vms)

    elif args.action == "kill-processes":
        # Kill all training processes
        manager.kill_all_processes(args.vms, dry_run=args.dry_run)

    elif args.action == "status":
        # Show current status
        manager.show_status(args.vms)


if __name__ == "__main__":
    print("usage")
    print("python3 enhanced_kill_manager.py --action kill-processes")

    main()
