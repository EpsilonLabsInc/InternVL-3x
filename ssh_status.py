#!/usr/bin/env python3
import subprocess
import concurrent.futures
import socket
from typing import List


def test_port_connectivity(
    source_vm: str, target_vm: str, port: int, ssh_user: str = "eric"
) -> dict:
    """Test if source_vm can connect to target_vm on the specified port"""
    test_cmd = f"timeout 5 bash -c '</dev/tcp/{target_vm}/{port}' 2>/dev/null && echo 'SUCCESS' || echo 'FAILED'"
    ssh_cmd = ["ssh", f"{ssh_user}@{source_vm}", test_cmd]

    try:
        result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=10)
        success = "SUCCESS" in result.stdout
        return {
            "source": source_vm,
            "target": target_vm,
            "port": port,
            "success": success,
            "output": result.stdout.strip(),
            "error": result.stderr.strip(),
        }
    except Exception as e:
        return {
            "source": source_vm,
            "target": target_vm,
            "port": port,
            "success": False,
            "output": "",
            "error": str(e),
        }


def test_distributed_connectivity(
    vms: List[str], master_port: int = 29500, ssh_user: str = "eric"
):
    """Test connectivity from all VMs to master"""
    master_vm = vms[0]  # Assuming first VM is master
    results = []

    print(f"Testing connectivity to master {master_vm}:{master_port}")
    print("=" * 60)

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = []

        # Test each VM's connectivity to master
        for vm in vms[1:]:  # Skip master itself
            future = executor.submit(
                test_port_connectivity, vm, master_vm, master_port, ssh_user
            )
            futures.append(future)

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)

            status = "✓" if result["success"] else "✗"
            print(f"{status} {result['source']} -> {result['target']}:{result['port']}")
            if not result["success"] and result["error"]:
                print(f"   Error: {result['error']}")

    failed_connections = [r for r in results if not r["success"]]
    if failed_connections:
        print(f"\n❌ {len(failed_connections)} connectivity issues found!")
        print(
            "These VMs cannot reach the master - this will cause distributed training to fail"
        )
    else:
        print(f"\n✅ All {len(results)} VMs can reach the master")

    return results


if __name__ == "__main__":
    # Your VM list
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

    # Test connectivity
    test_distributed_connectivity(all_vms[:5])  # Test first 5 VMs initially
