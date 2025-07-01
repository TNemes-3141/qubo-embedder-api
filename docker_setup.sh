#!/bin/bash
set -e

# Update package list and install prerequisites
echo "[+] Updating package list and installing ca-certificates and curl..."
sudo apt-get update
sudo apt-get install -y ca-certificates curl

# Create keyrings directory if it doesn't exist
echo "[+] Creating keyring directory for Docker..."
sudo install -m 0755 -d /etc/apt/keyrings

# Download and save Docker's GPG key
echo "[+] Downloading Docker's GPG key..."
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc

# Set permissions
echo "[+] Setting key permissions..."
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add Docker's repository to apt sources
echo "[+] Adding Docker's official repository..."
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
$(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Update apt package index
echo "[+] Updating package index..."
sudo apt-get update

echo "[✓] Docker repository setup complete."
