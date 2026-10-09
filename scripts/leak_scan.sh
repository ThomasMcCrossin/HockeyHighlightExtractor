#!/usr/bin/env bash
# Leak scan for the public tree: personal paths, e-mail addresses, keys and tokens, Drive
# folder ids, private-project names, and tracked media. Scans every file git would publish
# (tracked plus untracked-not-ignored) and the lines added since BASE (default origin/main).
#
#   scripts/leak_scan.sh [BASE]
#
# Exit 0 = clean, 1 = findings. Allow-listed on purpose: the public repo URL, the Resend
# sandbox sender, and example.* / *.invalid addresses.
set -uo pipefail
cd "$(dirname "$0")/.."
BASE="${1:-origin/main}"

PATTERNS=(
  '/home/[A-Za-z0-9_.-]+/'
  '/Users/[A-Za-z0-9_.-]+/'
  '[A-Za-z]:\\Users\\'
  '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'
  'sk-[A-Za-z0-9_-]{20,}'
  'gh[pousr]_[A-Za-z0-9]{30,}'
  'github_pat_[A-Za-z0-9_]{30,}'
  'AIza[0-9A-Za-z_-]{35}'
  'xox[abprs]-[A-Za-z0-9-]{10,}'
  'AKIA[0-9A-Z]{16}'
  'BEGIN [A-Z ]*PRIVATE KEY'
  '(api[_-]?key|secret|token|password)["'"'"' ]*[:=]["'"'"' ]*[A-Za-z0-9_/+=-]{24,}'
  'drive\.google\.com/[^ )"]*[A-Za-z0-9_-]{20,}'
  '/folders/[A-Za-z0-9_-]{20,}'
  'amherst-display|watch-rams|curlys|canteenhub|curlyshub|clarencehub|tailscale|\.ts\.net'
)
ALLOW='github\.com/ThomasMcCrossin/HockeyHighlightExtractor|onboarding@resend\.dev|@example\.|\.invalid|noreply@anthropic\.com|users\.noreply\.github\.com'

files=$(git ls-files --cached --others --exclude-standard | grep -v '^scripts/leak_scan.sh$' | grep -v 'package-lock.json')
fail=0
report() { echo "FINDING [$1] $2"; fail=1; }

# 1. every publishable text file
for pat in "${PATTERNS[@]}"; do
  while IFS= read -r hit; do
    [ -z "$hit" ] && continue
    echo "$hit" | grep -Eq "$ALLOW" && continue
    report tree "$hit"
  done < <(echo "$files" | xargs -d '\n' grep -nIE -- "$pat" 2>/dev/null)
done

# 2. lines added since BASE (catches content that a later edit removed from the tree but not from the diff)
if git rev-parse --verify -q "$BASE" >/dev/null; then
  for pat in "${PATTERNS[@]}"; do
    while IFS= read -r hit; do
      [ -z "$hit" ] && continue
      echo "$hit" | grep -Eq "$ALLOW" && continue
      report diff "$hit"
    done < <(git diff "$BASE" -U0 -- . ':!scripts/leak_scan.sh' ':!package-lock.json' | grep -E '^\+[^+]' | grep -E -- "$pat")
  done
fi

# 3. media and large files must not be tracked
while IFS= read -r f; do
  [ -z "$f" ] && continue
  case "$f" in
    skills/*/harness/*.ts) ;;   # a TypeScript extension, not an MPEG transport stream
    *.mp4|*.mkv|*.mov|*.ts|*.avi|*.webm|*.m4v|*.flv|*.wmv|*.wav|*.mp3) report media "$f" ;;
  esac
  size=$(stat -c %s "$f" 2>/dev/null || echo 0)
  [ "$size" -gt 1500000 ] && report large "$f ($size bytes)"
done <<< "$files"

if [ "$fail" -eq 0 ]; then
  echo "leak scan: clean ($(echo "$files" | wc -l) files, diff against $BASE)"
fi
exit "$fail"
