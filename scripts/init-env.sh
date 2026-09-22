#!/usr/bin/env bash
# Create .env from .env.example and fill in freshly generated secrets.
# Idempotent: never overwrites an existing .env, never regenerates existing secrets.
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -f .env ]]; then
  echo ".env already exists - leaving it untouched."
  exit 0
fi

random_hex() {
  if command -v openssl >/dev/null 2>&1; then
    openssl rand -hex 32
  else
    python3 -c 'import secrets; print(secrets.token_hex(32))'
  fi
}

umask 077  # .env holds secrets: owner-only permissions from the start
cp .env.example .env

for key in SECRET_KEY POSTGRES_PASSWORD; do
  value="$(random_hex)"
  # `|` delimiter is safe: the value is hex only. Replace only when the value is empty.
  sed -i.bak -E "s|^${key}=\$|${key}=${value}|" .env
done
rm -f .env.bak

echo "Created .env with generated SECRET_KEY and POSTGRES_PASSWORD."
