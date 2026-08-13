#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 SERVER_PID MERGED_MODEL OUTPUT.txt" >&2
    exit 2
fi

SERVER_PID=$1
MERGED_MODEL=$2
OUTPUT=$3

[[ $SERVER_PID =~ ^[1-9][0-9]*$ ]] || {
    echo "SERVER_PID must be a positive integer" >&2
    exit 2
}
[[ -r /proc/$SERVER_PID/status && -r /proc/$SERVER_PID/smaps ]] || {
    echo "server process is not readable: $SERVER_PID" >&2
    exit 1
}
[[ -f $MERGED_MODEL ]] || {
    echo "merged model is not a regular file: $MERGED_MODEL" >&2
    exit 1
}
mkdir -p -- "$(dirname -- "$OUTPUT")"

{
    printf 'captured_utc: '
    date -u +'%Y-%m-%dT%H:%M:%SZ'
    printf 'hostname: %s\n' "$(hostname)"
    printf 'kernel: '
    uname -srvm
    printf 'server_pid: %s\n' "$SERVER_PID"
    printf 'server_exe: %s\n' "$(readlink -- "/proc/$SERVER_PID/exe")"
    printf 'model_path: %s\n' "$MERGED_MODEL"
    printf 'model_bytes: %s\n' "$(stat -c %s -- "$MERGED_MODEL")"
    printf '\n/proc/meminfo:\n'
    awk '/^(MemTotal|MemFree|MemAvailable|Cached|SwapTotal|SwapFree):/' \
        /proc/meminfo
    printf '\nprocess status:\n'
    awk '/^(Name|State|VmPeak|VmSize|VmHWM|VmRSS|RssAnon|RssFile|VmSwap|Threads):/' \
        "/proc/$SERVER_PID/status"
    if [[ -r /proc/$SERVER_PID/smaps_rollup ]]; then
        printf '\nprocess smaps_rollup:\n'
        awk '/^(Rss|Pss|Pss_Anon|Pss_File|Private_Clean|Private_Dirty|Swap):/' \
            "/proc/$SERVER_PID/smaps_rollup"
    fi
    printf '\nGGUF mapping RSS:\n'
    awk -v model="$MERGED_MODEL" '
        /^[0-9a-f]+-[0-9a-f]+ / { matched = index($0, model) != 0; next }
        matched && /^Rss:/ { rss += $2 }
        END { printf "Rss: %d kB\n", rss + 0 }
    ' "/proc/$SERVER_PID/smaps"
    printf '\nNVIDIA state:\n'
    nvidia-smi --query-gpu=index,name,compute_cap,memory.total,memory.used,memory.free,utilization.gpu,driver_version \
        --format=csv,noheader
    printf '\nNVIDIA processes:\n'
    nvidia-smi --query-compute-apps=pid,process_name,used_memory \
        --format=csv,noheader || true
} >"$OUTPUT"

printf 'memory evidence: %s\n' "$OUTPUT"
