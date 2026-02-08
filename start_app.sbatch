#!/bin/bash
#SBATCH --job-name="w2d"
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=1:59:59
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=20G
#SBATCH --gres=gpu:L40S:1
#SBATCH --partition=L40Sday

mkdir -p logs

# Start SSH daemon on compute node so we can tunnel in
cat <<EOF > /tmp/sshd_config
Port 7665
HostKey ${HOME}/.ssh/id_ecdsa
PermitRootLogin no
PasswordAuthentication no
PubkeyAuthentication yes
AllowAgentForwarding yes
PidFile /tmp/sshd.pid
EOF
/usr/sbin/sshd -f /tmp/sshd_config

# Load env vars (HF_TOKEN etc.)
set -a
[ -f .env ] && source .env
set +a

# Run Streamlit inside Singularity (foreground, keeps job alive)
# Source code is bind-mounted from $(pwd) — no rebuild needed for code changes
singularity exec --nv \
    --env "HF_TOKEN=${HF_TOKEN:-}" \
    -B "$(pwd)":/app \
    -B "$(pwd)/cache":/app/cache \
    world2data.sif \
    python -m streamlit run /app/app.py \
        --server.port=8501 \
        --server.address=0.0.0.0 \
        --server.headless=true
