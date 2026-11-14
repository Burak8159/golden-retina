#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sdk_root="${ANDROID_SDK_ROOT:-${ANDROID_HOME:-$HOME/Library/Android/sdk}}"
avd_name="${1:-Medium_Phone_API_37.0}"
adb="$sdk_root/platform-tools/adb"
emulator="$sdk_root/emulator/emulator"
apk="$repo_root/android/app/build/outputs/apk/debug/app-debug.apk"
application_id="org.goldenhorn8159.goldenretina"

for binary in "$adb" "$emulator"; do
  if [[ ! -x "$binary" ]]; then
    printf 'Missing Android SDK tool: %s\nSet ANDROID_SDK_ROOT to your SDK location.\n' "$binary" >&2
    exit 1
  fi
done

if ! "$emulator" -list-avds | grep -Fxq "$avd_name"; then
  printf 'AVD not installed: %s\nInstalled AVDs:\n' "$avd_name" >&2
  "$emulator" -list-avds >&2
  exit 1
fi

export ANDROID_HOME="$sdk_root"
export ANDROID_SDK_ROOT="$sdk_root"
if [[ -f "$HOME/.android/adbkey" ]]; then
  export ADB_VENDOR_KEYS="$HOME/.android/adbkey"
fi

cd "$repo_root/android"
./gradlew :app:assembleDebug

find_target_serial() {
  while read -r serial state _; do
    [[ "$serial" == emulator-* && "$state" == device ]] || continue
    local running_avd
    running_avd="$("$adb" -s "$serial" shell getprop ro.boot.qemu.avd_name 2>/dev/null | tr -d '\r')"
    if [[ "$running_avd" == "$avd_name" ]]; then
      printf '%s' "$serial"
      return 0
    fi
  done < <("$adb" devices)
  return 1
}

if ! target_serial="$(find_target_serial)"; then
  "$emulator" -avd "$avd_name" -no-snapshot-load -no-boot-anim >/tmp/golden-retina-emulator.log 2>&1 &
fi

printf 'Waiting for Android emulator boot...\n'
for _ in $(seq 1 120); do
  target_serial="$(find_target_serial || true)"
  if [[ -n "$target_serial" && "$("$adb" -s "$target_serial" shell getprop sys.boot_completed 2>/dev/null | tr -d '\r')" == "1" ]]; then
    break
  fi
  sleep 1
done
if [[ -z "${target_serial:-}" || "$("$adb" -s "$target_serial" shell getprop sys.boot_completed 2>/dev/null | tr -d '\r')" != "1" ]]; then
  printf 'Emulator did not finish booting within 120 seconds. See /tmp/golden-retina-emulator.log\n' >&2
  exit 1
fi

"$adb" -s "$target_serial" install -r "$apk"
"$adb" -s "$target_serial" shell monkey -p "$application_id" 1
printf 'Launched %s on %s.\n' "$application_id" "$avd_name"
