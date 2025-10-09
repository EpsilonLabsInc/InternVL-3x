#!/usr/bin/env python3
import subprocess
import time
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional


class MultiVMJobManager:
    def __init__(self, config_file: Optional[str] = None):
        if config_file:
            with open(config_file, "r") as f:
                config = json.load(f)
                self.load_config(config)
        else:
            self.set_default_config()

    def set_default_config(self):
        all_vms = [
            "89.169.108.234",  # 0
            "192.168.0.0",  # 1
            "192.168.0.107",  # 2
            "192.168.0.187",  # 3
            "192.168.0.70",  # 4
            "192.168.0.233",  # 5
            "192.168.0.211",  # 6
            "192.168.0.236",  # 7
            "192.168.0.110",  # 8
            "192.168.0.66",  # 9
            "192.168.0.190",  # 10
            "192.168.0.210",  # 11
            "192.168.0.229",  # 12
            "192.168.0.228",  # 13
            "192.168.0.48",  # 14
            "192.168.0.6",  # 15
            "192.168.0.85",  # 16
            "192.168.0.68",  # 17
            "192.168.0.69",  # 18
            "192.168.0.129",  # 19
            "192.168.0.212",  # 20
            "192.168.0.100",  # 21
            "192.168.0.202",  # 22
            "192.168.0.50",  # 23
            "192.168.0.1",  # 24
            "192.168.0.206",  # 25
            "192.168.0.72",  # 26
            "192.168.0.71",  # 27
            "192.168.0.7",  # 28
            "192.168.0.92", # 29
            "192.168.0.97", # 30
        ]

        self.config = {
            "ssh_user": "eric",
            "nnodes": len(all_vms),
            "gpus": 8,
            "batch_size": 32,
            "per_device_batch_size": 4,
            "master_addr": "89.169.108.234",
            "master_port": 29500,
            # "script_name": "internvl3_train_multinode.sh",
            "script_name": "internvl3_train_multinode_stage1.sh",
            "ssh_key": "",  # Path to SSH key if needed
            "vms": all_vms,
            "tmux_session_prefix": "training_job",
            "launch_delay": 2,  # seconds between launches
            "reuse_sessions": True,  # New option to control session reuse
            "kill_existing_processes": True,  # Kill existing training processes before starting
        }

    def load_config(self, config: Dict):
        self.config = config

    def save_config(self, filename: str):
        with open(filename, "w") as f:
            json.dump(self.config, f, indent=2)

    def run_ssh_command(self, vm_ip: str, command: str) -> tuple:
        """Run SSH command and return (success, output, error)"""
        ssh_target = f"{self.config['ssh_user']}@{vm_ip}"
        ssh_cmd = ["ssh", ssh_target, command]

        try:
            result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=30)
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "SSH command timed out"
        except Exception as e:
            return False, "", str(e)

    def cleanup_existing_processes(self, vm_ip: str) -> bool:
        """Kill existing training processes and clean up"""
        if not self.config.get("kill_existing_processes", True):
            return True

        print(f"  Cleaning up existing processes on {vm_ip}...")

        # Kill existing training processes
        cleanup_cmds = [
            "pkill -f internvl3_train_multinode.sh 2>/dev/null || true",
            "pkill -f internvl 2>/dev/null || true",
            f"pkill -f '{self.config['script_name']}' 2>/dev/null || true",
        ]

        for cmd in cleanup_cmds:
            self.run_ssh_command(vm_ip, cmd)

        # Give processes time to terminate
        time.sleep(2)
        return True

    def handle_tmux_session(self, vm_ip: str, session_name: str) -> tuple:
        """Handle tmux session creation or reuse. Returns (success, is_new_session)"""

        # Check if session exists
        check_session_cmd = f"tmux has-session -t {session_name} 2>/dev/null"
        session_exists, _, _ = self.run_ssh_command(vm_ip, check_session_cmd)

        if session_exists and self.config.get("reuse_sessions", True):
            print(f"  Reusing existing tmux session: {session_name}")

            # Clear the session and stop any running commands
            clear_cmds = [
                f"tmux send-keys -t {session_name} C-c C-c",  # Stop current command
                f"tmux send-keys -t {session_name} 'clear' Enter",  # Clear screen
            ]

            for cmd in clear_cmds:
                self.run_ssh_command(vm_ip, cmd)

            time.sleep(1)
            return True, False

        elif session_exists:
            # Kill existing session if we don't want to reuse
            print(f"  Killing existing tmux session: {session_name}")
            kill_cmd = f"tmux kill-session -t {session_name} 2>/dev/null || true"
            self.run_ssh_command(vm_ip, kill_cmd)
            time.sleep(1)

        # Create new session
        create_session_cmd = f"tmux new-session -d -s {session_name}"
        success, stdout, stderr = self.run_ssh_command(vm_ip, create_session_cmd)

        if success:
            print(f"  Created new tmux session: {session_name}")
            return True, True
        else:
            print(f"  Failed to create tmux session: {stderr}")
            return False, False

    def start_job_on_vm(self, vm_ip: str, node_rank: int) -> Dict:
        """Start training job on a single VM"""
        log_name = f"master.log" if node_rank == 0 else f"worker{node_rank}.log"
        session_name = f"{self.config['tmux_session_prefix']}_{node_rank}"

        # Build environment variables
        env_vars = [
            f"NNODES={self.config['nnodes']}",
            f"NODE_RANK={node_rank}",
            f"GPUS={self.config['gpus']}",
            f"BATCH_SIZE={self.config['batch_size']}",
            f"PER_DEVICE_BATCH_SIZE={self.config['per_device_batch_size']}",
            f"MASTER_ADDR={self.config['master_addr']}",
            f"MASTER_PORT={self.config['master_port']}",
        ]

        # Build the training command
        working_dir = "~/projects/InternVL-3x/internvl_chat"

        wrapper_script = f"""#!/bin/bash
cd {working_dir}
{chr(10).join([f"export {var}" for var in env_vars])}
echo "Starting training with the following configuration:"
echo "NNODES={self.config["nnodes"]}, NODE_RANK={node_rank}, MASTER_ADDR={self.config["master_addr"]}"
echo "Log file: {log_name}"
echo "Timestamp: $(date)"
nohup bash {self.config["script_name"]} > {log_name} 2>&1 &
TRAIN_PID=$!
echo "Training job started with PID: $TRAIN_PID"
echo "You can monitor with: tail -f {log_name}"
"""

        print(f"Starting job on {vm_ip} (rank {node_rank})...")

        # Step 1: Clean up existing processes
        self.cleanup_existing_processes(vm_ip)

        # Step 2: Create wrapper script on the VM
        script_path = f"/tmp/train_wrapper_{node_rank}.sh"
        create_script_cmd = f'cat > {script_path} << "EOF"\n{wrapper_script}\nEOF'
        success1, stdout1, stderr1 = self.run_ssh_command(vm_ip, create_script_cmd)
        if not success1:
            return {
                "vm_ip": vm_ip,
                "node_rank": node_rank,
                "success": False,
                "session_name": session_name,
                "log_file": log_name,
                "error": f"Failed to create wrapper script: {stderr1}",
            }

        # Step 3: Make script executable
        success2, stdout2, stderr2 = self.run_ssh_command(
            vm_ip, f"chmod +x {script_path}"
        )
        if not success2:
            return {
                "vm_ip": vm_ip,
                "node_rank": node_rank,
                "success": False,
                "session_name": session_name,
                "log_file": log_name,
                "error": f"Failed to make script executable: {stderr2}",
            }

        # Step 4: Handle tmux session (create or reuse)
        session_success, is_new = self.handle_tmux_session(vm_ip, session_name)
        if not session_success:
            return {
                "vm_ip": vm_ip,
                "node_rank": node_rank,
                "success": False,
                "session_name": session_name,
                "log_file": log_name,
                "error": "Failed to create or reuse tmux session",
            }

        # Step 5: Send the wrapper script command to the session
        send_command_cmd = (
            f"tmux send-keys -t {session_name} 'bash {script_path}' Enter"
        )
        success4, stdout4, stderr4 = self.run_ssh_command(vm_ip, send_command_cmd)
        if not success4:
            # Clean up the session if command sending failed
            self.run_ssh_command(vm_ip, f"tmux kill-session -t {session_name}")
            return {
                "vm_ip": vm_ip,
                "node_rank": node_rank,
                "success": False,
                "session_name": session_name,
                "log_file": log_name,
                "error": f"Failed to send command to tmux session: {stderr4}",
            }

        # Step 6: Verify session and capture initial output
        time.sleep(3)  # Give it a moment to start
        verify_success, verify_stdout, _ = self.run_ssh_command(
            vm_ip, "tmux list-sessions"
        )
        session_running = verify_success and session_name in verify_stdout

        # Try to capture some output to see if the command is running
        if session_running:
            capture_success, capture_output, _ = self.run_ssh_command(
                vm_ip, f"tmux capture-pane -t {session_name} -p"
            )
            if capture_success and capture_output.strip():
                print(f"  Session output preview: {capture_output.strip()[-100:]}")

        result = {
            "vm_ip": vm_ip,
            "node_rank": node_rank,
            "success": session_running,
            "session_name": session_name,
            "log_file": log_name,
            "wrapper_script": script_path,
            "is_new_session": is_new,
            "error": None if session_running else "Session verification failed",
        }

        if session_running:
            print(f"✓ Successfully started job on {vm_ip}")
            print(f"  Tmux session: {session_name}")
            print(f"  Log file: {working_dir}/{log_name}")
        else:
            print(f"✗ Failed to verify job on {vm_ip}")

        return result

    def launch_all_jobs(self) -> List[Dict]:
        """Launch jobs on all VMs concurrently"""
        results = []

        print(f"Starting distributed training on {self.config['nnodes']} nodes...")
        print(f"Master: {self.config['master_addr']}:{self.config['master_port']}")
        print(f"Session reuse: {self.config.get('reuse_sessions', True)}")
        print("=" * 50)

        # Launch jobs with thread pool for concurrent execution
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_vm = {}

            for i, vm_ip in enumerate(self.config["vms"][: self.config["nnodes"]]):
                future = executor.submit(self.start_job_on_vm, vm_ip, i)
                future_to_vm[future] = (vm_ip, i)
                time.sleep(self.config["launch_delay"])  # Stagger launches

            # Collect results
            for future in as_completed(future_to_vm):
                result = future.result()
                results.append(result)

        return results

    def check_job_status(self, vm_ip: str, session_name: str) -> Dict:
        """Check if tmux session is still running"""
        cmd = f"tmux list-sessions | grep {session_name}"
        success, stdout, stderr = self.run_ssh_command(vm_ip, cmd)

        return {
            "vm_ip": vm_ip,
            "session_name": session_name,
            "running": success and session_name in stdout,
            "output": stdout,
        }

    def monitor_jobs(self, results: List[Dict]):
        """Monitor job status"""
        print("\n" + "=" * 50)
        print("Monitoring jobs...")
        print("=" * 50)

        for result in results:
            if result["success"]:
                status = self.check_job_status(result["vm_ip"], result["session_name"])
                status_str = "RUNNING" if status["running"] else "STOPPED"
                print(f"{result['vm_ip']} (rank {result['node_rank']}): {status_str}")

    def kill_all_jobs(self, results: List[Dict]):
        """Kill all training jobs"""
        print("Killing all jobs...")
        for result in results:
            vm_ip = result["vm_ip"]
            session_name = result["session_name"]

            # Kill tmux session
            cmd = f"tmux kill-session -t {session_name} 2>/dev/null || true"
            success, _, _ = self.run_ssh_command(vm_ip, cmd)

            # Also kill any remaining training processes
            cleanup_cmds = [
                "pkill -f internvl3_train_multinode.sh 2>/dev/null || true",
                "pkill -f internvl 2>/dev/null || true",
                f"pkill -f '{self.config['script_name']}' 2>/dev/null || true",
            ]

            for cleanup_cmd in cleanup_cmds:
                self.run_ssh_command(vm_ip, cleanup_cmd)

            status = "✓" if success else "✗"
            print(f"{status} {vm_ip}: {session_name}")


def main():
    parser = argparse.ArgumentParser(description="Multi-VM Training Job Manager")
    parser.add_argument("--config", help="Config file path (JSON)")
    parser.add_argument(
        "--action",
        choices=["launch", "monitor", "kill"],
        default="launch",
        help="Action to perform",
    )
    parser.add_argument("--save-config", help="Save default config to file")
    parser.add_argument(
        "--no-reuse", action="store_true", help="Don't reuse existing tmux sessions"
    )

    args = parser.parse_args()

    manager = MultiVMJobManager(args.config)

    # Override reuse setting if specified
    if args.no_reuse:
        manager.config["reuse_sessions"] = False

    if args.save_config:
        manager.save_config(args.save_config)
        print(f"Config saved to {args.save_config}")
        return

    if args.action == "launch":
        results = manager.launch_all_jobs()

        # Save results for later monitoring
        with open("job_results.json", "w") as f:
            json.dump(results, f, indent=2)

        print("\n" + "=" * 50)
        print("Job launch complete!")
        print(f"Results saved to job_results.json")
        print("\nTo monitor: python3 job_manager.py --action monitor")
        print("To kill all: python3 job_manager.py --action kill")

    elif args.action == "monitor":
        try:
            with open("job_results.json", "r") as f:
                results = json.load(f)
            manager.monitor_jobs(results)
        except FileNotFoundError:
            print("No job results found. Run launch first.")

    elif args.action == "kill":
        try:
            with open("job_results.json", "r") as f:
                results = json.load(f)
            manager.kill_all_jobs(results)
        except FileNotFoundError:
            print("No job results found.")


if __name__ == "__main__":
    print("usage")
    print("  python3 job_manager.py --action launch")
    print("  python3 job_manager.py --action monitor")
    print("  python3 job_manager.py --action kill")
    print("  python3 job_manager.py --action launch --no-reuse  # Force new sessions")

    main()
