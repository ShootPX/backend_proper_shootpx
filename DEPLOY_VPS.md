# Deploying to a Hostinger VPS (Ubuntu 22.04/24.04)

Layout: nginx (443) -> uvicorn (127.0.0.1:8000) + a separate arq worker.
Postgres, Redis and Supabase stay external (as today); the VPS only needs outbound access.

## 1. Server prep
```bash
ssh root@<vps-ip>
apt update && apt upgrade -y
apt install -y python3 python3-venv python3-pip nginx certbot python3-certbot-nginx git ufw libpq-dev
adduser --system --group --home /opt/shootpx shootpx
ufw allow OpenSSH && ufw allow 'Nginx Full' && ufw enable
```
Point your API domain's DNS **A record** at the VPS IP.

## 2. Code + dependencies
```bash
mkdir -p /opt/shootpx && cd /opt/shootpx
git clone <your-repo-url> backend && cd backend
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```
If `pip` fails on a Windows-only package, remove that line (nothing here should need it).

## 3. Secrets
```bash
cp .env.production.example .env && nano .env      # fill every value
# upload the Firebase key from your PC (never committed to git):
#   scp firebase-service-account.json root@<vps-ip>:/opt/shootpx/backend/
chown -R shootpx:shootpx /opt/shootpx
chmod 600 .env firebase-service-account.json
```
Key settings: `ENV=production`, `TRUST_PROXY=true`, `CORS_ORIGINS` = your real frontend origins,
`PUBLIC_BACKEND_URL` = `https://<api-domain>` (fal.ai/Razorpay webhooks must reach it).

Apply any pending SQL in `migrations/` to your database (e.g. `psql "$DATABASE_URL" -f migrations/<file>.sql`).

## 4. Services
```bash
cp deploy/shootpx-api.service deploy/shootpx-worker.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now shootpx-api shootpx-worker
systemctl status shootpx-api shootpx-worker
```

## 5. nginx + HTTPS
```bash
cp deploy/nginx-shootpx.conf /etc/nginx/sites-available/shootpx
nano /etc/nginx/sites-available/shootpx            # set server_name
ln -s /etc/nginx/sites-available/shootpx /etc/nginx/sites-enabled/
nginx -t && systemctl reload nginx
certbot --nginx -d <api-domain>
```
Verify: `curl https://<api-domain>/health`

## 6. Third-party dashboards
- Razorpay webhook URL -> `https://<api-domain>/billing/webhook` (fal.ai calls `/webhooks/fal` automatically via `PUBLIC_BACKEND_URL`)
- Firebase console -> Authentication -> Authorized domains: add the frontend domain.
- Frontend: point its API base URL at `https://<api-domain>`.

## Operations
- Logs: `journalctl -u shootpx-api -f` / `journalctl -u shootpx-worker -f`
- Update: `cd /opt/shootpx/backend && git pull && venv/bin/pip install -r requirements.txt && systemctl restart shootpx-api shootpx-worker`
- Run only ONE worker instance (cron jobs are scheduled inside it).
