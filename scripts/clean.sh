#!/usr/bin/env bash

set -e

# Colors for output
GREEN='\033[0;32m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

# Remove logs, checkpoints, weights, rollouts
rm -rf logs checkpoints weights rollouts
log_info "Cleaned up!"