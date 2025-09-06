#!/usr/bin/env python3

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# List of IP addresses
ips = [
    "192.168.0.0",  # 1
    "192.168.0.107",  # 2
    # "192.168.0.187",  # 3
    # "192.168.0.70",  # 4
    # "192.168.0.113",  # 5
    # "192.168.0.211",  # 6
    # "192.168.0.137",  # 7
    # "192.168.0.110",  # 8
    # "192.168.0.66",  # 9
    # "192.168.0.136",  # 10
    # "192.168.0.170",  # 11
    # "192.168.0.229",  # 12
    # "192.168.0.228",  # 13
    # "192.168.0.48",  # 14
    # "192.168.0.6",  # 15
    # "192.168.0.93",  # 16
    # "192.168.0.68",  # 17
    # "192.168.0.69",  # 18
    # "192.168.0.129",  # 19
    # "192.168.0.212",  # 20
    # "192.168.0.100",  # 21
    # "192.168.0.202",  # 22
    # "192.168.0.50",  # 23
    # "192.168.0.33",  # 24
    # "192.168.0.146",  # 25
    # "192.168.0.72",  # 26
    # "192.168.0.71",  # 27
]

# Configuration
USERNAME = "eric"  # Change this to your username
SSH_KEY = ""  # Path to SSH key if needed, e.g., "/home/user/.ssh/id_rsa"
TIMEOUT = 10  # Connection timeout in seconds


def ssh_to_vm(ip, vm_number):
    """SSH to a VM and list ~/.ssh contents"""

    # Build SSH command
    ssh_cmd = [
        "ssh",
        "-o",
        "ConnectTimeout={}".format(TIMEOUT),
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "BatchMode=yes",  # Avoid password prompts
    ]

    if SSH_KEY:
        ssh_cmd.extend(["-i", SSH_KEY])

    ssh_cmd.extend(["{}@{}".format(USERNAME, ip), "ls -la ~/.ssh"])

    try:
        result = subprocess.run(
            ssh_cmd, capture_output=True, text=True, timeout=TIMEOUT + 5
        )

        if result.returncode == 0:
            return {
                "ip": ip,
                "vm_number": vm_number,
                "status": "success",
                "output": result.stdout.strip(),
                "error": "",
            }
        else:
            return {
                "ip": ip,
                "vm_number": vm_number,
                "status": "failed",
                "output": "",
                "error": result.stderr.strip(),
            }

    except subprocess.TimeoutExpired:
        return {
            "ip": ip,
            "vm_number": vm_number,
            "status": "timeout",
            "output": "",
            "error": "Connection timed out",
        }
    except Exception as e:
        return {
            "ip": ip,
            "vm_number": vm_number,
            "status": "error",
            "output": "",
            "error": str(e),
        }


def main():
    print("Starting SSH connections to all VMs...")
    print(f"Username: {USERNAME}")
    print(f"Total VMs: {len(ips)}")
    print("=" * 50)

    # You can choose between sequential or parallel execution
    use_parallel = input("Use parallel connections? (y/n): ").lower().startswith("y")

    if use_parallel:
        # Parallel execution (faster but may overwhelm network)
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_vm = {
                executor.submit(ssh_to_vm, ip, i + 1): (ip, i + 1)
                for i, ip in enumerate(ips)
            }

            results = []
            for future in as_completed(future_to_vm):
                result = future.result()
                results.append(result)

        # Sort results by VM number for display
        results.sort(key=lambda x: x["vm_number"])
    else:
        # Sequential execution
        results = []
        for i, ip in enumerate(ips):
            result = ssh_to_vm(ip, i + 1)
            results.append(result)
            print(f"Processed VM {i + 1}/{len(ips)}: {ip}")

    # Display results
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    successful = 0
    failed = 0

    for result in results:
        print(f"\nVM #{result['vm_number']} - {result['ip']}")
        print("-" * 40)

        if result["status"] == "success":
            print("✓ SUCCESS")
            print("~/.ssh contents:")
            print(result["output"])
            successful += 1
        else:
            print("✗ FAILED")
            print(f"Status: {result['status']}")
            print(f"Error: {result['error']}")
            failed += 1

    print(f"\n" + "=" * 60)
    print(f"FINAL SUMMARY: {successful} successful, {failed} failed")
    print("=" * 60)


if __name__ == "__main__":
    main()
