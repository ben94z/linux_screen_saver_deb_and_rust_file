# BEN94Z 30-second Linux screensaver

Builds a `.deb` from the supplied BEN94Z Rust source. **This ZIP is a source/build kit, not a precompiled `.deb`.** Requires internet access on first Cargo build.

## Ubuntu installation

```bash
sudo apt update
sudo apt install -y cargo rustc build-essential pkg-config libasound2-dev libudev-dev libgl1-mesa-dev libx11-dev libxi-dev libxcursor-dev libxrandr-dev libxinerama-dev libdbus-1-dev xprintidle
chmod +x build-deb.sh
./build-deb.sh
sudo apt install ./BEN94Z_30s_Screensaver_1.1.0_$(dpkg --print-architecture).deb
```

Log out and log back in to start the autostart monitor, or launch it immediately with:

```bash
nohup python3 /usr/lib/ben94z/ben94z-idle.py >/tmp/ben94z-idle.log 2>&1 &
```

The monitor polls GNOME Mutter's IdleMonitor via session D-Bus (works on supported GNOME Wayland/X11 sessions), then falls back to `xprintidle` on X11. It launches BEN94Z fullscreen at 30 seconds and terminates it on activity. If no compatible idle service is available, it won't start the screensaver. It does not bypass GNOME's screen lock; adjust GNOME's existing blank/lock timeout separately if needed.

Manual preview: `ben94z --screensaver` (press Escape to exit). Normal application: `ben94z`.

To disable autostart, copy `/etc/xdg/autostart/ben94z-idle.desktop` to `~/.config/autostart/` and add `Hidden=true`.

**Note:** This package has not been compiled/tested in this environment (Cargo toolchain unavailable here). The `Cargo.toml` dependencies are inferred from the source; adjust versions if compilation reports API mismatches. On Debian-based distributions, dependency names may differ. Existing installed BEN94Z packages will be upgraded/replaced by this version.
