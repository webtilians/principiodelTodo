#!/usr/bin/env bash
# INFINITO's memory over HTTP (src/infinito3/remote.py) on the LAIN server, for ChatGPT and other
# remote MCP clients, next to LAIN (8000), DH Trails (8100), the Laboratory (8200), RecipAI (8300)
# and the demo (8400). Run as root:
#   bash remote/deploy/vps/setup.sh
#   REMOTE_BRANCH=feat/x bash remote/deploy/vps/setup.sh   (remembered in /etc/infinito-remote/branch)
# It copies itself to /opt/infinito-remote-deploy and carries on from there. Afterwards
# infinito-remote-update.timer follows that branch on GitHub every 2 minutes by itself.
# Running this again is safe: people and their memories are kept.
set -euo pipefail

DATA=/var/lib/infinito-remote
CONF=/etc/infinito-remote
DEPLOY=/opt/infinito-remote-deploy
KIT=$(cd "$(dirname "$0")" && pwd -P)

step() { printf '\n== %s\n' "$*"; }

[ "$(id -u)" -eq 0 ] || { echo "Ejecuta este script como root." >&2; exit 1; }
cd /
if [ "$KIT" != "$DEPLOY" ]; then
  rm -rf "$DEPLOY.new"
  cp -r "$KIT" "$DEPLOY.new"
  rm -rf "$DEPLOY"
  mv "$DEPLOY.new" "$DEPLOY"
  chmod 755 "$DEPLOY/setup.sh" "$DEPLOY"/bin/*
  exec bash "$DEPLOY/setup.sh" "$@"
fi
install -d -o root -g root -m 755 "$CONF"
branch=${REMOTE_BRANCH:-$(cat "$CONF/branch" 2>/dev/null || echo master)}
host=${REMOTE_SITE:-$(cat "$CONF/host" 2>/dev/null || true)}
if [ -z "$host" ]; then
  # Free HTTPS name next to LAIN's: 1.2.3.4 -> infinito.1-2-3-4.sslip.io resolves back to this IP.
  ip=$(ip -4 route get 1.1.1.1 | awk '{for (i = 1; i < NF; i++) if ($i == "src") { print $(i + 1); exit }}')
  host=infinito.${ip//./-}.sslip.io
fi
echo "$branch" > "$CONF/branch"
echo "$host" > "$CONF/host"

step "Paquetes del sistema"
if ! dpkg -s python3-venv git curl >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -q
  apt-get install -y -q --no-upgrade python3-venv git curl
fi

step "Usuario y carpetas"
id infinito-remote >/dev/null 2>&1 || useradd --system --home-dir "$DATA" --shell /usr/sbin/nologin infinito-remote
install -d -o infinito-remote -g infinito-remote -m 700 "$DATA"

step "Servicio"
install -m 755 "$KIT"/bin/* /usr/local/sbin/
install -m 644 "$KIT"/systemd/* /etc/systemd/system/
systemctl daemon-reload
systemctl enable -q infinito-remote.service
# Installs the code in its venv and starts the service.
infinito-remote-update --force

step "Caddy ($host)"
install -d -m 755 /etc/caddy/sites.d
sed "s/__HOST__/$host/" "$KIT/infinito-remote.caddy" > /etc/caddy/sites.d/infinito-remote.caddy
grep -qxF 'import /etc/caddy/sites.d/*.caddy' /etc/caddy/Caddyfile \
  || printf '\n# Other sites on this server, one file each.\nimport /etc/caddy/sites.d/*.caddy\n' >> /etc/caddy/Caddyfile
if ! check=$(caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile 2>&1); then
  echo "$check" >&2
  rm -f /etc/caddy/sites.d/infinito-remote.caddy
  echo "Caddy no acepta la configuración: se ha retirado el sitio y los demás siguen igual." >&2
  exit 1
fi
systemctl reload caddy

step "Actualización automática y copias de seguridad"
systemctl enable -q --now infinito-remote-update.timer
systemctl enable -q --now infinito-remote-backup.timer

step "Comprobación"
ok=""
for _ in $(seq 1 30); do
  if curl -fs --max-time 5 -o /dev/null "https://$host/health"; then ok=1; break; fi
  sleep 3
done
echo
if [ -n "$ok" ]; then
  echo "LISTO: https://$host/  (cada persona necesita su dirección: infinito-remote-person add <nombre>)"
else
  echo "El servicio arrancó, pero https://$host aún no responde."
  echo "Revisa: systemctl status infinito-remote caddy; journalctl -u infinito-remote -n 50"
fi
infinito-remote-status
