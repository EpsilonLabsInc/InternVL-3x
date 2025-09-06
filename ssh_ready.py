#!/usr/bin/env python3

import subprocess
import sys
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# List of VM IP addresses
vm_ips = [
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

# Configuration
USERNAME = "eric"
HOST_IP = "89.169.108.234"  # Your host IP
TIMEOUT = 10


def run_ssh_command(target_ip, command, timeout=TIMEOUT, use_key_auth=True):
    """Run SSH command with optional key authentication"""
    ssh_cmd = [
        "ssh",
        "-o",
        f"ConnectTimeout={timeout}",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "BatchMode=yes" if use_key_auth else "BatchMode=no",
        f"{USERNAME}@{target_ip}",
        command,
    ]

    try:
        result = subprocess.run(
            ssh_cmd, capture_output=True, text=True, timeout=timeout + 5
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": "Connection timeout",
            "returncode": -1,
        }
    except Exception as e:
        return {"success": False, "stdout": "", "stderr": str(e), "returncode": -1}


def test_host_to_vm(vm_ip, vm_number):
    """Test SSH connection from host to VM"""
    print(f"Testing host → VM #{vm_number} ({vm_ip})... ", end="", flush=True)

    # Simple connectivity test
    result = run_ssh_command(vm_ip, "echo 'Host-to-VM connection successful'")

    if result["success"]:
        print("✓ SUCCESS")
        return True
    else:
        print(f"✗ FAILED ({result['stderr'][:50]})")
        return False


def test_vm_to_host(vm_ip, vm_number):
    """Test SSH connection from VM back to host"""
    print(f"Testing VM #{vm_number} ({vm_ip}) → host... ", end="", flush=True)

    # Command to run on VM: SSH back to host
    test_command = f"ssh -o ConnectTimeout={TIMEOUT} -o StrictHostKeyChecking=no -o BatchMode=yes {USERNAME}@{HOST_IP} 'echo \"VM-to-host connection successful from $(hostname)\"'"

    result = run_ssh_command(vm_ip, test_command)

    if result["success"]:
        print(f"✓ SUCCESS - {result['stdout']}")
        return True
    else:
        print(f"✗ FAILED ({result['stderr'][:50]})")
        return False


def test_vm_comprehensive(vm_ip, vm_number):
    """Run comprehensive test for a single VM"""
    print(f"\n{'=' * 60}")
    print(f"VM #{vm_number} - {vm_ip}")
    print(f"{'=' * 60}")

    # Test 1: Host to VM
    host_to_vm_success = test_host_to_vm(vm_ip, vm_number)

    # Test 2: VM to Host (only if host-to-vm works)
    if host_to_vm_success:
        time.sleep(0.5)  # Small delay between tests
        vm_to_host_success = test_vm_to_host(vm_ip, vm_number)
    else:
        print(f"Skipping VM-to-host test (host-to-VM failed)")
        vm_to_host_success = False

    return {
        "vm_number": vm_number,
        "ip": vm_ip,
        "host_to_vm": host_to_vm_success,
        "vm_to_host": vm_to_host_success,
        "both_directions": host_to_vm_success and vm_to_host_success,
    }


def run_tests_sequential():
    """Run tests sequentially (slower but easier to read)"""
    print("Running tests sequentially...")
    results = []

    for i, vm_ip in enumerate(vm_ips, 1):
        try:
            result = test_vm_comprehensive(vm_ip, i)
            results.append(result)
            time.sleep(0.5)  # Small delay between VMs
        except KeyboardInterrupt:
            print("\n\nTests cancelled by user.")
            break
        except Exception as e:
            print(f"\nError testing VM #{i} ({vm_ip}): {e}")
            results.append(
                {
                    "vm_number": i,
                    "ip": vm_ip,
                    "host_to_vm": False,
                    "vm_to_host": False,
                    "both_directions": False,
                }
            )

    return results


def run_tests_parallel():
    """Run tests in parallel (faster but mixed output)"""
    print("Running tests in parallel...")
    results = []

    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit all test jobs
        future_to_vm = {
            executor.submit(test_vm_comprehensive, vm_ip, i + 1): (vm_ip, i + 1)
            for i, vm_ip in enumerate(vm_ips)
        }

        # Collect results as they complete
        for future in as_completed(future_to_vm):
            try:
                result = future.result()
                results.append(result)
                print(f"Completed VM #{result['vm_number']}")
            except Exception as e:
                vm_ip, vm_num = future_to_vm[future]
                print(f"Error testing VM #{vm_num} ({vm_ip}): {e}")
                results.append(
                    {
                        "vm_number": vm_num,
                        "ip": vm_ip,
                        "host_to_vm": False,
                        "vm_to_host": False,
                        "both_directions": False,
                    }
                )

    # Sort results by VM number
    results.sort(key=lambda x: x["vm_number"])
    return results


def print_summary(results):
    """Print detailed summary of all test results"""
    print(f"\n{'=' * 80}")
    print("COMPREHENSIVE TEST SUMMARY")
    print(f"{'=' * 80}")

    # Count results
    total_vms = len(results)
    host_to_vm_success = sum(1 for r in results if r["host_to_vm"])
    vm_to_host_success = sum(1 for r in results if r["vm_to_host"])
    bidirectional_success = sum(1 for r in results if r["both_directions"])

    print(f"Total VMs tested: {total_vms}")
    print(f"Host → VM successful: {host_to_vm_success}/{total_vms}")
    print(f"VM → Host successful: {vm_to_host_success}/{total_vms}")
    print(f"Bidirectional successful: {bidirectional_success}/{total_vms}")
    print()

    # Detailed results
    print("Detailed Results:")
    print("-" * 80)
    print(
        f"{'VM#':<4} {'IP Address':<15} {'Host→VM':<10} {'VM→Host':<10} {'Status':<15}"
    )
    print("-" * 80)

    for result in results:
        vm_num = result["vm_number"]
        ip = result["ip"]
        h2v = "✓" if result["host_to_vm"] else "✗"
        v2h = "✓" if result["vm_to_host"] else "✗"

        if result["both_directions"]:
            status = "✓ FULL SUCCESS"
        elif result["host_to_vm"]:
            status = "⚠ PARTIAL"
        else:
            status = "✗ FAILED"

        print(f"{vm_num:<4} {ip:<15} {h2v:<10} {v2h:<10} {status:<15}")

    # Problem VMs
    failed_vms = [r for r in results if not r["host_to_vm"]]
    partial_vms = [r for r in results if r["host_to_vm"] and not r["vm_to_host"]]

    if failed_vms:
        print(f"\n❌ VMs with connection failures:")
        for vm in failed_vms:
            print(f"   VM #{vm['vm_number']} ({vm['ip']}) - Cannot connect from host")

    if partial_vms:
        print(f"\n⚠️  VMs with partial connectivity:")
        for vm in partial_vms:
            print(
                f"   VM #{vm['vm_number']} ({vm['ip']}) - Can connect to VM, but VM cannot connect back to host"
            )

    if bidirectional_success == total_vms:
        print(f"\n🎉 ALL VMs have bidirectional SSH connectivity!")

    print(f"\n{'=' * 80}")


def main():
    print("=" * 80)
    print("SSH CONNECTIVITY TEST")
    print("=" * 80)
    print(f"Host IP: {HOST_IP}")
    print(f"Username: {USERNAME}")
    print(f"VMs to test: {len(vm_ips)}")
    print(f"Test timeout: {TIMEOUT} seconds")
    print()

    # Choose test mode
    mode = input("Run tests in (s)equential or (p)arallel mode? [s/p]: ").lower()
    use_parallel = mode.startswith("p")

    print(f"\nStarting tests at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 80)

    start_time = time.time()

    try:
        if use_parallel:
            results = run_tests_parallel()
        else:
            results = run_tests_sequential()
    except KeyboardInterrupt:
        print("\n\nTesting interrupted by user.")
        return

    end_time = time.time()

    print(f"\nTesting completed in {end_time - start_time:.1f} seconds")

    # Print comprehensive summary
    print_summary(results)


if __name__ == "__main__":
    main()
