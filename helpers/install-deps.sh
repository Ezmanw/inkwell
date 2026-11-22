#!/bin/bash
# Inkwell build-dependency installer. Runs as root via pkexec.
# usage: install-deps.sh <cuda|vulkan|rocm|cpu> <driver-cuda-version|-> <distro-id|-> <distro-version|->
# Set SIMULATE=1 to only print what apt would do.
set -euo pipefail
backend=${1:-cpu}; cuda_max=${2:--}; distro=${3:--}; ver=${4:--}
APT=(apt-get install -y --no-install-recommends)
[ "${SIMULATE:-0}" = 1 ] && APT=(apt-get install -s --no-install-recommends)
export DEBIAN_FRONTEND=noninteractive

case "$backend" in cuda|vulkan|rocm|cpu) ;; *) echo "unknown backend: $backend" >&2; exit 2;; esac

pkgs=(git cmake ninja-build build-essential ca-certificates)
case "$backend" in
  vulkan) pkgs+=(libvulkan-dev glslc spirv-headers) ;;
  rocm)
    if ls -d /opt/rocm*/bin/hipcc /opt/rocm*/*/bin/hipcc >/dev/null 2>&1; then
      # AMD's ROCm is already installed. Don't add the distro's older ROCm next to it: CMake then mixes
      # the two (distro hip-config + AMD's split layout) and the build fails.
      echo "AMD ROCm found in /opt/rocm; not installing the distro's ROCm packages."
    else
      echo "Note: the distro's ROCm can be too old for the newest cards (e.g. RX 9000 / gfx12 needs ROCm 6.4+)."
      pkgs+=(hipcc libhipblas-dev librocblas-dev rocminfo)
    fi ;;
esac

if [ "$backend" = cuda ]; then
  # Pick the newest CUDA toolkit the installed driver supports, from NVIDIA's own repo.
  if ! ls /usr/local/cuda*/bin/nvcc >/dev/null 2>&1 || ! ls /usr/local/cuda*/targets/*/include/nvml.h >/dev/null 2>&1; then
    case "$distro-$ver" in
      ubuntu-24.04|pop-24.04|linuxmint-22*|zorin-18*) repo=ubuntu2404 ;;
      ubuntu-22.04|pop-22.04|linuxmint-21*|zorin-17*) repo=ubuntu2204 ;;
      debian-12*) repo=debian12 ;;
      debian-13*) repo=debian13 ;;
      *) . /etc/os-release
         case "${UBUNTU_CODENAME:-$VERSION_CODENAME}" in
           noble) repo=ubuntu2404 ;; jammy) repo=ubuntu2204 ;; bookworm) repo=debian12 ;; trixie) repo=debian13 ;;
           *) echo "Can't pick an NVIDIA CUDA repo for $distro $ver. Install the CUDA toolkit yourself." >&2; exit 3 ;;
         esac ;;
    esac
    if [ ! -f /usr/share/keyrings/cuda-archive-keyring.gpg ]; then
      tmp=$(mktemp -d)
      curl -fsSL -o "$tmp/k.deb" "https://developer.download.nvidia.com/compute/cuda/repos/$repo/x86_64/cuda-keyring_1.1-1_all.deb" \
        || wget -qO "$tmp/k.deb" "https://developer.download.nvidia.com/compute/cuda/repos/$repo/x86_64/cuda-keyring_1.1-1_all.deb"
      [ "${SIMULATE:-0}" = 1 ] || dpkg -i "$tmp/k.deb"
      rm -rf "$tmp"
    fi
    # Low priority: CUDA packages install on request, but NVIDIA's repo never replaces your graphics driver.
    [ "${SIMULATE:-0}" = 1 ] || printf 'Package: *\nPin: origin developer.download.nvidia.com\nPin-Priority: 100\n' \
      > /etc/apt/preferences.d/cuda-repository-pin-600
    [ "${SIMULATE:-0}" = 1 ] || apt-get update
    # driver supports e.g. 13.2 -> newest cuda-nvcc-13-X with X <= 2
    major=${cuda_max%%.*}; minor=${cuda_max#*.}; minor=${minor%%.*}
    best=""
    for v in $(apt-cache search --names-only '^cuda-nvcc-[0-9]+-[0-9]+$' | awk '{print $1}' | sed 's/cuda-nvcc-//' | sort -t- -k1,1n -k2,2n); do
      M=${v%-*}; m=${v#*-}
      if [ "$cuda_max" = "-" ] || [ "$M" -lt "$major" ] || { [ "$M" -eq "$major" ] && [ "$m" -le "$minor" ]; }; then best=$v; fi
    done
    [ -n "$best" ] || { echo "No CUDA toolkit found that your driver (CUDA $cuda_max) supports. Update the NVIDIA driver." >&2; exit 4; }
    echo "Installing CUDA toolkit $best (driver supports $cuda_max)"
    pkgs+=(cuda-nvcc-$best cuda-cudart-dev-$best libcublas-dev-$best cuda-nvrtc-dev-$best cuda-cccl-$best cuda-nvml-dev-$best)
  fi
fi

[ "${SIMULATE:-0}" = 1 ] || apt-get update
"${APT[@]}" "${pkgs[@]}"
echo "Done."
