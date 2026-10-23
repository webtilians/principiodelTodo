#!/usr/bin/env bash
# INFINITO Laboratory (lab_portal/ of webtilians/principiodelTodo) on the LAIN server, next to
# LAIN (127.0.0.1:8000) and DH Trails (127.0.0.1:8100). Run as root, from any copy of this folder:
#   bash lab_portal/deploy/vps/setup.sh
#   LAB_BRANCH=experiment/x bash lab_portal/deploy/vps/setup.sh   (remembered in /etc/infinito-lab/branch)
# It copies itself to /opt/infinito-lab-deploy and carries on from there. Afterwards
# infinito-lab-update.timer follows that branch on GitHub every 2 minutes by itself.
# Running this again is safe: the key and the experiment results are kept.
set -euo pipefail

DATA=/var/lib/infinito-lab
CONF=/etc/infinito-lab
DEPLOY=/opt/infinito-lab-deploy
KIT=$(cd "$(dirname "$0")" && pwd -P)

step() { printf '\n== %s\n' "$*"; }

[ "$(id -u)" -eq 0 ] || { echo "Ejecuta este script como root." >&2; exit 1; }
cd /
# Never run from /opt/infinito-lab: the update below rewrites that checkout while bash reads this file.
if [ "$KIT" != "$DEPLOY" ]; then
  rm -rf "$DEPLOY.new"
  cp -r "$KIT" "$DEPLOY.new"
  rm -rf "$DEPLOY"
  mv "$DEPLOY.new" "$DEPLOY"
  chmod 755 "$DEPLOY/setup.sh" "$DEPLOY"/bin/*
  exec bash "$DEPLOY/setup.sh" "$@"
fi
install -d -o root -g root -m 755 "$CONF"
branch=${LAB_BRANCH:-$(cat "$CONF/branch" 2>/dev/null || echo experiment/f8-laboratory-portal)}
host=${LAB_SITE:-$(cat "$CONF/host" 2>/dev/null || true)}
if [ -z "$host" ]; then
  # Free HTTPS name next to LAIN's: 1.2.3.4 -> lab.1-2-3-4.sslip.io resolves back to this IP.
  ip=$(ip -4 route get 1.1.1.1 | awk '{for (i = 1; i < NF; i++) if ($i == "src") { print $(i + 1); exit }}')
  host=lab.${ip//./-}.sslip.io
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
id infinito-lab >/dev/null 2>&1 || useradd --system --home-dir "$DATA" --shell /usr/sbin/nologin infinito-lab
install -d -o infinito-lab -g infinito-lab -m 700 "$DATA" "$DATA/runs"

step "Clave del laboratorio"
if [ ! -s "$CONF/lab.env" ]; then
  install -m 600 /dev/null "$CONF/lab.env"
  cat > "$CONF/lab.env" <<EOF
# INFINITO Laboratory, written once by setup.sh. Show the key: infinito-lab-token
LAB_API_TOKEN=$(python3 -c 'import secrets; print(secrets.token_urlsafe(36))')
# Longest run in seconds (60 to 86400). After editing: systemctl restart infinito-lab
LAB_MAX_SECONDS=14400
EOF
else
  echo "Se conserva $CONF/lab.env"
fi

step "Servicio"
install -m 755 "$KIT"/bin/* /usr/local/sbin/
install -m 644 "$KIT"/systemd/* /etc/systemd/system/
systemctl daemon-reload
systemctl enable -q infinito-lab.service
# Fetches the code, installs it in its venv and starts the service (waits if an experiment runs).
infinito-lab-update --force

step "Caddy ($host)"
install -d -m 755 /etc/caddy/sites.d
sed "s/__HOST__/$host/" "$KIT/infinito-lab.caddy" > /etc/caddy/sites.d/infinito-lab.caddy
grep -qxF 'import /etc/caddy/sites.d/*.caddy' /etc/caddy/Caddyfile \
  || printf '\n# Other sites on this server, one file each.\nimport /etc/caddy/sites.d/*.caddy\n' >> /etc/caddy/Caddyfile
if ! check=$(caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile 2>&1); then
  echo "$check" >&2
  rm -f /etc/caddy/sites.d/infinito-lab.caddy
  echo "Caddy no acepta la configuración: se ha retirado el sitio del laboratorio y LAIN sigue igual." >&2
  exit 1
fi
systemctl reload caddy

step "Actualización automática"
systemctl enable -q --now infinito-lab-update.timer

step "Comprobación"
ok=""
for _ in $(seq 1 30); do
  if curl -fs --max-time 5 -o /dev/null "https://$host/"; then ok=1; break; fi
  sleep 3
done
echo
if [ -n "$ok" ]; then
  echo "LISTO: https://$host/"
else
  echo "El laboratorio arrancó, pero https://$host aún no responde."
  echo "Revisa: systemctl status infinito-lab caddy; journalctl -u infinito-lab -n 50  (el certificado puede tardar un minuto)"
fi
infinito-lab-status
