#!/usr/bin/env bash
# Разворачивает worktree-ы под волну задач, чтобы агенты работали параллельно
# и не мешали друг другу.
#
#   ./setup_worktrees.sh 1     # A1..A5
#   ./setup_worktrees.sh 2     # B1, B2
#   ./setup_worktrees.sh 3     # C1..C4
#   ./setup_worktrees.sh 4     # D1..D4
#   ./setup_worktrees.sh clean # удалить все worktree задач
#
# Запускать из корня репозитория. Базовая ветка — integration.

set -euo pipefail

BASE_BRANCH="${BASE_BRANCH:-integration}"
WT_ROOT="${WT_ROOT:-..}"

declare -A WAVES=(
  [1]="A1-splits A2-stats A3-schema A4-logprobs A5-hygiene"
  [2]="B1-protocol B2-score-cli"
  [3]="C1-stacking C2-encoder C3-m3-axes C4-m6-grounding"
  [4]="D1-perchunk D2-gepa D3-notebooks D4-reporting"
)

if [[ "${1:-}" == "clean" ]]; then
  for w in "${WAVES[@]}"; do
    for t in $w; do
      id="${t%%-*}"
      git worktree remove --force "$WT_ROOT/wt-$id" 2>/dev/null || true
      git branch -D "task/$t" 2>/dev/null || true
    done
  done
  git worktree prune
  echo "worktree-ы задач удалены"
  exit 0
fi

WAVE="${1:?укажи номер волны: 1, 2, 3, 4 или clean}"
TASKS="${WAVES[$WAVE]:?неизвестная волна: $WAVE}"

git fetch origin "$BASE_BRANCH" 2>/dev/null || true

echo "База: $BASE_BRANCH"
echo

for t in $TASKS; do
  id="${t%%-*}"
  dir="$WT_ROOT/wt-$id"
  if [[ -d "$dir" ]]; then
    echo "  $id  уже существует: $dir  (пропуск)"
    continue
  fi
  git worktree add "$dir" -b "task/$t" "$BASE_BRANCH" >/dev/null
  echo "  $id  -> $dir   ветка task/$t"
done

echo
echo "Промпт для каждого агента (запускать в соответствующем каталоге):"
echo
for t in $TASKS; do
  id="${t%%-*}"
  cat <<EOF
  cd $WT_ROOT/wt-$id && claude "Прочитай AGENTS.md и docs/handoff/tasks/$id.md и выполни задачу $id целиком. Меняй только файлы из раздела «Владеет». Прогони критерии приёмки и открой PR в ветку $BASE_BRANCH."

EOF
done

echo "После слияния волны:"
echo "  git checkout $BASE_BRANCH && git pull && make check"
echo "  ./setup_worktrees.sh clean && ./setup_worktrees.sh $((WAVE + 1))"
