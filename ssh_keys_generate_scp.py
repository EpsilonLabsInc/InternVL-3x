#!/usr/bin/env python3

import subprocess
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# List of IP addresses
ips = [
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

USERNAME = "eric"
TIMEOUT = 10


def run_ssh_command(ip, command, timeout=TIMEOUT):
    """Run a command via SSH and return the result"""
    ssh_cmd = [
        "ssh",
        "-o",
        f"ConnectTimeout={timeout}",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "BatchMode=yes",
        f"{USERNAME}@{ip}",
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


def check_vm_connectivity(ip):
    """Check if VM is reachable via SSH"""
    result = run_ssh_command(ip, "echo 'Connected'", timeout=5)
    return result["success"]


def check_ssh_keys_exist(ip):
    """Check if SSH key pair exists on the VM"""
    result = run_ssh_command(
        ip,
        "test -f ~/.ssh/id_ed25519 && test -f ~/.ssh/id_ed25519.pub && echo 'exists'",
    )
    return result["success"] and result["stdout"] == "exists"


def generate_ssh_key(ip):
    """Generate SSH key pair on the VM"""
    result = run_ssh_command(
        ip, "ssh-keygen -t ed25519 -C 'eric' -f ~/.ssh/id_ed25519 -N ''"
    )
    return result["success"]


def get_public_key(ip):
    """Get the public key from the VM"""
    result = run_ssh_command(ip, "cat ~/.ssh/id_ed25519.pub")
    if result["success"]:
        return result["stdout"]
    return None


def process_vm(ip, vm_number):
    """Process a single VM: check keys, generate if needed, collect public key"""
    print(f"\nVM #{vm_number} - {ip}")
    print("-" * 40)

    # Check connectivity
    if not check_vm_connectivity(ip):
        print("✗ Cannot connect to VM - skipping")
        return None

    print("✓ Connected successfully")

    # Check if keys exist
    if check_ssh_keys_exist(ip):
        print("✓ SSH key pair already exists")
        key_status = "existing"
    else:
        print("✗ No SSH key pair found")
        print("  Generating new SSH key pair...")

        if generate_ssh_key(ip):
            print("✓ SSH key pair generated successfully")
            key_status = "generated"
        else:
            print("✗ Failed to generate SSH key pair")
            return None

    # Get the public key
    pub_key = get_public_key(ip)
    if pub_key:
        print(f"✓ Public key collected ({key_status})")
        return {
            "ip": ip,
            "vm_number": vm_number,
            "public_key": pub_key,
            "status": key_status,
        }
    else:
        print("✗ Failed to read public key")
        return None


def add_keys_to_authorized_keys(collected_keys):
    """Add collected public keys to local authorized_keys file"""
    home_dir = Path.home()
    ssh_dir = home_dir / ".ssh"
    authorized_keys_file = ssh_dir / "authorized_keys"

    # Create .ssh directory if it doesn't exist
    ssh_dir.mkdir(mode=0o700, exist_ok=True)

    # Backup existing authorized_keys
    if authorized_keys_file.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = authorized_keys_file.with_suffix(f".backup.{timestamp}")
        import shutil

        shutil.copy2(authorized_keys_file, backup_file)
        print(f"✓ Backed up existing authorized_keys to: {backup_file}")

    # Read existing keys to avoid duplicates
    existing_keys = set()
    if authorized_keys_file.exists():
        try:
            with open(authorized_keys_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        existing_keys.add(line)
        except Exception as e:
            print(f"Warning: Could not read existing authorized_keys: {e}")

    # Add new keys (append mode to preserve existing content)
    new_keys_added = 0
    duplicate_keys = 0

    with open(authorized_keys_file, "a") as f:
        for key_info in collected_keys:
            vm_number = key_info["vm_number"]
            ip = key_info["ip"]
            pub_key = key_info["public_key"]
            status = key_info["status"]

            # Check for duplicates
            if pub_key in existing_keys:
                print(f"- Key already exists for VM #{vm_number} ({ip})")
                duplicate_keys += 1
            else:
                # Add comment and key
                f.write(f"# VM #{vm_number} - {ip} ({status} key)\n")
                f.write(f"{pub_key}\n")
                existing_keys.add(pub_key)
                print(f"✓ Added key for VM #{vm_number} ({ip})")
                new_keys_added += 1

    # Set correct permissions
    authorized_keys_file.chmod(0o600)

    return new_keys_added, duplicate_keys


def main():
    print("=" * 60)
    print("SSH Key Generation and Collection Script")
    print("=" * 60)
    print(f"Target VMs: {len(ips)}")
    print(f"Username: {USERNAME}")
    print("")

    print("Step 1: Processing all VMs...")
    print("=" * 30)

    collected_keys = []
    successful_vms = 0
    failed_vms = 0

    for i, ip in enumerate(ips, 1):
        try:
            result = process_vm(ip, i)
            if result:
                collected_keys.append(result)
                successful_vms += 1
            else:
                failed_vms += 1
        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user.")
            sys.exit(1)
        except Exception as e:
            print(f"✗ Unexpected error processing VM #{i} ({ip}): {e}")
            failed_vms += 1

        # Small delay between VMs
        time.sleep(0.5)

    print(f"\nStep 1 Summary:")
    print(f"✓ Successful: {successful_vms}")
    print(f"✗ Failed: {failed_vms}")
    print(f"📊 Public keys collected: {len(collected_keys)}")

    if not collected_keys:
        print("\nNo public keys were collected. Exiting.")
        return

    print(f"\nStep 2: Adding keys to authorized_keys...")
    print("=" * 40)

    try:
        new_keys, duplicate_keys = add_keys_to_authorized_keys(collected_keys)

        print(f"\nStep 2 Summary:")
        print(f"✓ New keys added: {new_keys}")
        print(f"- Duplicate keys skipped: {duplicate_keys}")

        # Get local IP for instructions
        try:
            result = subprocess.run(["hostname", "-I"], capture_output=True, text=True)
            local_ip = (
                result.stdout.strip().split()[0]
                if result.returncode == 0
                else "YOUR_HOST_IP"
            )
        except:
            local_ip = "YOUR_HOST_IP"

        print(f"\n" + "=" * 60)
        print("SETUP COMPLETE!")
        print("=" * 60)
        print("You can now SSH from any VM to this host using:")
        print(f"ssh {USERNAME}@{local_ip}")
        print(f"\nAuthorized keys file: {Path.home()}/.ssh/authorized_keys")

    except Exception as e:
        print(f"✗ Error adding keys to authorized_keys: {e}")
        return


if __name__ == "__main__":
    main()
