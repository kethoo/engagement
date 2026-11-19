#!/usr/bin/env bash
# One-screen status of the cluster runs. From the laptop:
#   ssh -t micm-cluster 'watch -n 10 ~/wassa-engagement/scripts/status.sh'
# Reads only processes, logs and runs/*/results.json — starts no computation.
cd "$(dirname "$0")/.."

# --- what is running right now (from the process list, so it works for any launcher/log)
JOB=$(pgrep -af "python -m engagement" | grep -v pgrep | head -1)
if [ -n "$JOB" ]; then
  PID=${JOB%% *}
  HEAD=$(grep -oP -- '--head \K\S+' <<<"$JOB" || echo "?")
  SETS=$(grep -oP -- '--set \K\S+' <<<"$JOB" | tr '\n' ' ')
  STAGE=$(grep -oP 'engagement \K(prepare|features|train)' <<<"$JOB")
  echo "RUNNING: $STAGE  head=$HEAD  ${SETS:+variant: $SETS} (pid $PID, $(ps -o etime= -p "$PID" | xargs))"
else
  echo "RUNNING: nothing"
fi

# newest log = the active (or last) run
LOG=${LOG:-$(ls -t logs/*.log 2>/dev/null | head -1)}
[ -n "$LOG" ] && echo "log: $LOG (last written $(( $(date +%s) - $(stat -c %Y "$LOG") ))s ago)"

python3 - "$LOG" "$HEAD" <<'EOF'
import glob, json, re, sys
log = open(sys.argv[1], errors='replace').read() if sys.argv[1] else ''
tail = log.split('wrote runs/')[-1]  # the head currently being trained
m = re.findall(r'\[(\d+)/(\d+)\] ', tail)
if m and 'saved cache' not in tail:
    print(f'progress: extracting features {m[-1][0]}/{m[-1][1]} articles')
elif '== article_cv' in tail:
    folds = len(re.findall(r'repeat \d+ fold \d+: chose', tail))
    print(f'progress: CV on train  [{"#" * folds}{"." * (25 - folds)}] {folds}/25 folds')
elif '== holdout' in tail:
    print('progress: holdout (val selection -> test)')
errs = [l for l in log.splitlines() if re.search(r'Traceback|Error|Killed|out of memory', l)]
if errs:
    print('!! error in log:', errs[-1][:150])

def label(path):
    name = path.split('/')[1]
    head = re.search(r'_(tfidf_ridge|ridge|linear|mlp)_\d{8}-\d{6}$', name).group(1)
    tags = []
    if 'exclude_test_articles_from_train=false' in name:
        tags.append('keep-test-arts')
    if 'kind=raw_mean' in name:
        tags.append('raw-mean-target')
    return ' + '.join(tags) or 'main', head

rows = []
for f in glob.glob('runs/*/results.json'):
    if 'superseded' in f:
        continue
    r = json.load(open(f))
    variant, head = label(f)
    cv = r.get('article_cv', {}).get('metrics')
    h = r.get('holdout', {}).get('metrics')
    rows.append((variant, head, cv, h))

order = {'main': 0, 'raw-mean-target': 1, 'keep-test-arts': 2}
heads = {'ridge': 0, 'mlp': 1, 'linear': 2, 'tfidf_ridge': 3}
print(f'\n{"variant":16s} {"head":11s} | {"CV train (adj)":>15s} | {"TEST essay [90% CI]":>23s} | {"TEST article":>12s} | chosen on val')
last = None
for variant, head, cv, h in sorted(rows, key=lambda x: (order.get(x[0], 9), heads[x[1]])):
    if last is not None and variant != last:
        print('-' * 100)
    last = variant
    cvs = f'{cv["article_adjusted_avg"]["mean"]:+.3f} ± {cv["article_adjusted_avg"]["sd"]:.3f}' if cv else '-'
    if h:
        e, a = h['test']['essay_avg'], h['test']['article_raw_avg']
        ts = f'{e["value"]:+.3f} [{e["ci90"][0]:+.2f}, {e["ci90"][1]:+.2f}]'
        print(f'{variant:16s} {head:11s} | {cvs:>15s} | {ts:>23s} | {a["value"]:+12.3f} | {h["chosen"]}')
    else:
        print(f'{variant:16s} {head:11s} | {cvs:>15s} | {"-":>23s} | {"-":>12s} |')
EOF

echo
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null | sed 's/^/GPU /'
