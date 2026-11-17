#!/usr/bin/env bash
# One-screen status of the cluster run. From the laptop:
#   ssh -t micm-cluster 'watch -n 10 ~/wassa-engagement/scripts/status.sh'
# Reads only the log, the pid file and runs/*/results.json — starts no computation.
cd "$(dirname "$0")/.."
LOG=${LOG:-logs/baseline.log}
PID=$(cat logs/baseline.pid 2>/dev/null)

if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
  echo "RUN: running (pid $PID, $(ps -o etime= -p "$PID" | xargs) elapsed)"
elif grep -q ALL_DONE "$LOG" 2>/dev/null; then
  echo "RUN: finished"
else
  echo "RUN: not running"
fi
echo "log last written $(( $(date +%s) - $(stat -c %Y "$LOG") ))s ago"

python3 - "$LOG" <<'EOF'
import glob, json, re, sys
log = open(sys.argv[1], errors='replace').read()
heads = ['tfidf_ridge', 'ridge', 'linear', 'mlp']
done = re.findall(r'wrote runs/\S+?_(tfidf_ridge|ridge|linear|mlp)_\d', log)
stage = 'features' if 'saved cache' not in log and 'features cached' not in log else None
if stage:
    m = re.findall(r'\[(\d+)/(\d+)\] ', log)
    print(f'stage: features {m[-1][0]}/{m[-1][1]} articles' if m else 'stage: loading model')
elif len(done) < len(heads):
    cur = heads[len(done)]
    tail = log.split('wrote runs/')[-1] if done else log
    folds = len(re.findall(r'repeat \d+ fold \d+: chose', tail))
    phase = 'article_cv' if '== article_cv' in tail else 'holdout'
    bar = '#' * folds + '.' * (25 - folds)
    print(f'stage: head {cur} ({len(done) + 1}/4)  {phase}  cv folds [{bar}] {folds}/25')
    last = re.findall(r'repeat \d+: article_adjusted_avg=\S+', tail)
    if last:
        print('  last repeat:', last[-1])
if re.search(r'Traceback|Error|Killed|out of memory', log):
    print('!! error in log:', [l for l in log.splitlines() if re.search(r'Traceback|Error|Killed|out of memory', l)][-1][:150])

print(f'\n{"run":30s} | {"CV(train) article_adj":>21s} | {"TEST essay_avg [90% CI]":>25s} {"article":>7s}  chosen')
for f in sorted(glob.glob('runs/*/results.json')):
    if 'superseded' in f:
        continue
    r = json.load(open(f))
    run = re.sub(r'^runs/baseline_article_only_?|_\d{8}-\d{6}/results.json$', '', f)
    cv = r.get('article_cv', {}).get('metrics')
    cvs = f'{cv["article_adjusted_avg"]["mean"]:+.3f} ± {cv["article_adjusted_avg"]["sd"]:.3f}' if cv else '-'
    h = r.get('holdout', {}).get('metrics')
    hs = '-'
    if h:
        e, a = h['test']['essay_avg'], h['test']['article_raw_avg']
        hs = f'{e["value"]:+.3f} [{e["ci90"][0]:+.2f}, {e["ci90"][1]:+.2f}] {a["value"]:+7.3f}  {h["chosen"]}'
    print(f'{run[:30]:30s} | {cvs:>21s} | {hs}')
EOF

echo
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null | sed 's/^/GPU /'
