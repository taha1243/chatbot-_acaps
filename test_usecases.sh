#!/bin/bash
# Systematic test of chatbot use cases against /chat (JSON)

API="http://localhost:8080/chat"
OUT="test_results"
mkdir -p "$OUT"

run_test() {
  local id="$1"
  local label="$2"
  local question="$3"
  local outfile="$OUT/${id}.json"

  echo ">>> [$id] $label"
  echo "    Q: $question"

  start=$(date +%s)
  http_code=$(curl -s -X POST "$API" \
    -H "Content-Type: application/json" \
    -d "{\"question\":\"$question\"}" \
    --max-time 300 \
    -o "$outfile" \
    -w "%{http_code}")
  end=$(date +%s)
  duration=$((end - start))

  if [ "$http_code" != "200" ]; then
    echo "    [HTTP $http_code in ${duration}s] FAIL"
    echo ""
    return
  fi

  # Extract fields with python (file-based, no shell quoting issues)
  py - <<PYEOF "$outfile" "$id" "$label" "$duration"
import json, sys, re
path, _id, label, duration = sys.argv[1:5]
with open(path, encoding='utf-8') as f:
    d = json.load(f)
ans = d.get('answer', '')
cits = d.get('citations', [])
conf = d.get('confidence', 0)
meta = d.get('metadata', {})
intent = meta.get('intent', '-')
src = meta.get('source', '-')
ar_chars = sum(1 for c in ans if '؀' <= c <= 'ۿ')
total_chars = max(len(ans), 1)
ar_ratio = ar_chars / total_chars
lang = 'AR' if ar_ratio > 0.3 else ('FR' if ar_ratio < 0.05 else 'MIX')
preview = ans.strip().replace('\n', ' / ')[:120]
print(f"    [{duration}s | {len(cits)} cites | conf={conf:.2f} | intent={intent} | src={src} | lang={lang}]")
print(f"    >> {preview}...")
PYEOF
  echo ""
}

run_test 01 "FR Submit"        "Comment soumettre une reclamation sur le portail ?"
run_test 02 "FR Track"         "Comment suivre ma reclamation ?"
run_test 03 "FR Close"         "Comment clôturer ou réouvrir une réclamation ?"
run_test 04 "FR Satisfaction"  "Comment acceder au questionnaire de satisfaction ?"
run_test 05 "AR Submit"        "كيف أقدم شكاية على البوابة ؟"
run_test 06 "AR Track"         "كيف أتابع شكايتي ؟"
run_test 07 "Greeting"         "Bonjour"
run_test 08 "Out-of-scope"     "Quelle est la capitale du Maroc ?"

echo "=== Done ==="
