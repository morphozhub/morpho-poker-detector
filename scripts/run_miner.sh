#!/bin/bash
# morpho-poker-detector miner startup (pm2)
set -e
NETUID="${NETUID:-126}"
WALLET_NAME="${WALLET_NAME:?set WALLET_NAME}"
HOTKEY="${HOTKEY:?set HOTKEY}"
NETWORK="${NETWORK:-finney}"
AXON_PORT="${AXON_PORT:-8091}"
PM2_NAME="${PM2_NAME:-morpho_miner}"
ALLOWED_VALIDATOR_HOTKEYS="${ALLOWED_VALIDATOR_HOTKEYS:-}"

cd "$(dirname "$0")/.."
ARGS=(--netuid "$NETUID" --wallet.name "$WALLET_NAME" --wallet.hotkey "$HOTKEY"
      --subtensor.network "$NETWORK" --axon.port "$AXON_PORT" --logging.debug)
if [ -n "$ALLOWED_VALIDATOR_HOTKEYS" ]; then
  read -r -a VH <<< "$ALLOWED_VALIDATOR_HOTKEYS"
  ARGS+=(--blacklist.allowed_validator_hotkeys "${VH[@]}")
else
  ARGS+=(--blacklist.force_validator_permit)
fi
pm2 delete "$PM2_NAME" 2>/dev/null || true
pm2 start miner/miner.py --name "$PM2_NAME" --interpreter python3 -- "${ARGS[@]}"
pm2 save
echo "started $PM2_NAME (netuid=$NETUID, port=$AXON_PORT)"
