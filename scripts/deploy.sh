#!/usr/bin/env bash
set -euo pipefail

HOST="webmestres@solidaires-etudiant-e-s.org"
APP="solifloute"
YNH_REPO="https://github.com/Solidaires-Etudiant-e-s/solifloute_ynh"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=== Solifloute Deploy Script ==="
echo

BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$BRANCH" != "main" ]; then
  echo "Error: must be on main branch (currently on $BRANCH)"
  exit 1
fi

if [ -n "$(git status --porcelain)" ]; then
  echo "Error: working directory has uncommitted changes"
  exit 1
fi

echo "Deploying $APP to $HOST..."
echo "Running: sudo yunohost app upgrade $APP -u $YNH_REPO"
echo

ssh -t "$HOST" "sudo yunohost app upgrade $APP -u $YNH_REPO"

echo
echo "Done!"
