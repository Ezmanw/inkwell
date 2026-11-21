#!/bin/bash
# Build dist/inkwell_<version>_all.deb from this checkout.
set -euo pipefail
cd "$(dirname "$0")/.."
VER=$(cat VERSION)
ROOT=build/deb/inkwell_${VER}_all
rm -rf "$ROOT"; mkdir -p "$ROOT/DEBIAN" dist
install -d "$ROOT/usr/lib/inkwell/inkwell" "$ROOT/usr/lib/inkwell/helpers" "$ROOT/usr/bin" \
  "$ROOT/usr/share/applications" "$ROOT/usr/share/icons/hicolor/scalable/apps" "$ROOT/usr/share/metainfo" \
  "$ROOT/usr/share/polkit-1/actions" "$ROOT/usr/share/doc/inkwell"
install -m644 inkwell/*.py "$ROOT/usr/lib/inkwell/inkwell/"
install -m755 helpers/install-deps.sh "$ROOT/usr/lib/inkwell/helpers/"
install -m755 bin/inkwell "$ROOT/usr/bin/"
install -m644 data/io.github.Ezmanw.Inkwell.desktop "$ROOT/usr/share/applications/"
install -m644 data/io.github.Ezmanw.Inkwell.svg "$ROOT/usr/share/icons/hicolor/scalable/apps/"
install -m644 data/io.github.Ezmanw.Inkwell.metainfo.xml "$ROOT/usr/share/metainfo/"
install -m644 data/io.github.Ezmanw.Inkwell.policy "$ROOT/usr/share/polkit-1/actions/"
install -m644 LICENSE "$ROOT/usr/share/doc/inkwell/copyright"
SIZE=$(du -sk "$ROOT/usr" | cut -f1)
cat > "$ROOT/DEBIAN/control" <<CTL
Package: inkwell
Version: $VER
Section: utils
Priority: optional
Architecture: all
Installed-Size: $SIZE
Maintainer: Ezmanw <https://github.com/Ezmanw>
Depends: python3 (>= 3.10), python3-gi, gir1.2-gtk-4.0 (>= 4.12), gir1.2-adw-1 (>= 1.5), python3-yaml, python3-qrcode, git, cmake, build-essential, pciutils, pkexec | policykit-1
Recommends: ninja-build, ca-certificates, curl
Homepage: https://github.com/Ezmanw/inkwell
Description: Run Inkling 2 Mini on your own PC
 Inkwell builds llama.cpp for your graphics card (CUDA, Vulkan, ROCm or CPU),
 downloads the Inkling 2 Mini model size that fits, and gives you a native
 GTK chat app with image input, a phone API and dsh integration.
CTL
cat > "$ROOT/DEBIAN/postinst" <<'PI'
#!/bin/sh
set -e
command -v gtk-update-icon-cache >/dev/null && gtk-update-icon-cache -q /usr/share/icons/hicolor || true
command -v update-desktop-database >/dev/null && update-desktop-database -q /usr/share/applications || true
PI
chmod 755 "$ROOT/DEBIAN/postinst"
dpkg-deb --root-owner-group -Zxz --build "$ROOT" "dist/inkwell_${VER}_all.deb"
echo "Built dist/inkwell_${VER}_all.deb"
