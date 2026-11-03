#!/usr/bin/env bash
# INFINITO public demo (demo/ of webtilians/principiodelTodo) on the LAIN server, next to
# LAIN (8000), DH Trails (8100), the INFINITO Laboratory (8200) and RecipAI (8300). Run as root:
#   bash demo/deploy/vps/setup.sh
#   DEMO_BRANCH=feat/x bash demo/deploy/vps/setup.sh   (remembered in /etc/infinito-demo/branch)
# It copies itself to /opt/infinito-demo-deploy and carries on from there. Afterwards
# infinito-demo-update.timer follows that branch on GitHub every 2 minutes by itself.
# Running this again is safe: the settings and the OpenAI key are kept.
set -euo pipefail

DATA=/var/lib/infinito-demo
CONF=/etc/infinito-demo
DEPLOY=/opt/infinito-demo-deploy
KIT=$(cd "$(dirname "$0")" && pwd -P)

step() { printf '\n== %s\n' "$*"; }

[ "$(id -u)" -eq 0 ] || { echo "Ejecuta este script como root." >&2; exit 1; }
cd /
# Never run from /opt/infinito-demo: the update below rewrites that checkout while bash reads this file.
if [ "$KIT" != "$DEPLOY" ]; then
  rm -rf "$DEPLOY.new"
  cp -r "$KIT" "$DEPLOY.new"
  rm -rf "$DEPLOY"
  mv "$DEPLOY.new" "$DEPLOY"
  chmod 755 "$DEPLOY/setup.sh" "$DEPLOY"/bin/*
  exec bash "$DEPLOY/setup.sh" "$@"
fi
install -d -o root -g root -m 755 "$CONF"
branch=${DEMO_BRANCH:-$(cat "$CONF/branch" 2>/dev/null || echo master)}
host=${DEMO_SITE:-$(cat "$CONF/host" 2>/dev/null || true)}
if [ -z "$host" ]; then
  # Free HTTPS name next to LAIN's: 1.2.3.4 -> demo.1-2-3-4.sslip.io resolves back to this IP.
  ip=$(ip -4 route get 1.1.1.1 | awk '{for (i = 1; i < NF; i++) if ($i == "src") { print $(i + 1); exit }}')
  host=demo.${ip//./-}.sslip.io
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
id infinito-demo >/dev/null 2>&1 || useradd --system --home-dir "$DATA" --shell /usr/sbin/nologin infinito-demo
install -d -o infinito-demo -g infinito-demo -m 700 "$DATA"

step "Ajustes"
if [ ! -s "$CONF/demo.env" ]; then
  install -m 600 /dev/null "$CONF/demo.env"
  cat > "$CONF/demo.env" <<'EOF'
# INFINITO demo, written once by setup.sh. After editing: systemctl restart infinito-demo
# The OpenAI key goes in with infinito-demo-set-key (never by hand in a shared screen).
# Most the demo may spend per UTC day, in USD; when reached it says so until midnight UTC.
DEMO_DAILY_BUDGET=1.0
# Messages per visitor, and per connection per hour.
DEMO_MESSAGES_PER_VISITOR=30
DEMO_MESSAGES_PER_IP_HOUR=90
EOF
else
  echo "Se conserva $CONF/demo.env"
fi

step "Servicio"
install -m 755 "$KIT"/bin/* /usr/local/sbin/
install -m 644 "$KIT"/systemd/* /etc/systemd/system/
systemctl daemon-reload
systemctl enable -q infinito-demo.service
# Fetches the code, installs it in its venv and starts the service.
infinito-demo-update --force

step "Caddy ($host)"
install -d -m 755 /etc/caddy/sites.d
sed "s/__HOST__/$host/" "$KIT/infinito-demo.caddy" > /etc/caddy/sites.d/infinito-demo.caddy
grep -qxF 'import /etc/caddy/sites.d/*.caddy' /etc/caddy/Caddyfile \
  || printf '\n# Other sites on this server, one file each.\nimport /etc/caddy/sites.d/*.caddy\n' >> /etc/caddy/Caddyfile
if ! check=$(caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile 2>&1); then
  echo "$check" >&2
  rm -f /etc/caddy/sites.d/infinito-demo.caddy
  echo "Caddy no acepta la configuración: se ha retirado el sitio de la demo y LAIN sigue igual." >&2
  exit 1
fi
systemctl reload caddy

step "Actualización automática"
systemctl enable -q --now infinito-demo-update.timer

step "Comprobación"
ok=""
for _ in $(seq 1 30); do
  if curl -fs --max-time 5 -o /dev/null "https://$host/healthz"; then ok=1; break; fi
  sleep 3
done
echo
if [ -n "$ok" ]; then
  echo "LISTO: https://$host/"
else
  echo "La demo arrancó, pero https://$host aún no responde."
  echo "Revisa: systemctl status infinito-demo caddy; journalctl -u infinito-demo -n 50  (el certificado puede tardar un minuto)"
fi
infinito-demo-status
