# Push the current code to the GCE VM and rebuild. deploy/.env and deploy/hf_token already live on the VM.
# usage: powershell -File deploy\deploy.ps1   (needs gcloud authenticated, CLOUDSDK_CONFIG set if non-default)
param(
    [string]$Instance = "va-server",
    [string]$Zone = "us-central1-a",
    [string]$Gcloud = "gcloud"
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$tgz = Join-Path $env:TEMP "va_app.tgz"
Push-Location $root
tar -czf $tgz --exclude=work --exclude=jobs --exclude=__pycache__ --exclude=.git --exclude=deploy/.env --exclude=deploy/hf_token .
Pop-Location
& $Gcloud compute scp $tgz "${Instance}:/tmp/va_app.tgz" --zone=$Zone --quiet --strict-host-key-checking=no
# One line: gcloud.cmd passes the command through cmd.exe, which breaks multi-line strings.
$remote = "set -e; cd ~/app; " +
    "find . -mindepth 1 -maxdepth 1 ! -name deploy -exec rm -rf {} +; " +
    "find deploy -mindepth 1 ! -name .env ! -name hf_token -exec rm -rf {} +; " +
    "tar -xzf /tmp/va_app.tgz -C ~/app; rm /tmp/va_app.tgz; " +
    "cd deploy; sudo docker compose up -d --build > ~/build.log 2>&1 < /dev/null; " +
    "sudo docker image prune -f > /dev/null; tail -3 ~/build.log"
& $Gcloud compute ssh $Instance --zone=$Zone --quiet --strict-host-key-checking=no --command=$remote
