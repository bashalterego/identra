#!/usr/bin/env bash
# Quick IPC Communication Test
# Usage: ./tools/quick_ipc_test.sh

set -e

echo "🧪 Quick IPC Test for Identra"
echo "=============================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Build both components
echo -e "${YELLOW}📦 Building components...${NC}"
cargo build --bin vault-daemon --release --quiet
cargo build --example test_vault_ipc --quiet

# Kill any existing vault-daemon
pkill -9 vault-daemon 2>/dev/null || true
rm -f /tmp/identra-vault.sock

# Start vault-daemon
echo -e "${YELLOW}🔐 Starting vault-daemon...${NC}"
./target/release/vault-daemon > /tmp/vault_test.log 2>&1 &
VAULT_PID=$!
sleep 2

# Check if running
if kill -0 $VAULT_PID 2>/dev/null; then
    echo -e "${GREEN}✅ vault-daemon running (PID: $VAULT_PID)${NC}"
else
    echo "❌ vault-daemon failed to start"
    cat /tmp/vault_test.log
    exit 1
fi

# Run integration test
echo -e "${YELLOW}🧪 Running integration test...${NC}"
./target/debug/examples/test_vault_ipc

# Cleanup
echo -e "${YELLOW}🧹 Cleaning up...${NC}"
kill -9 $VAULT_PID 2>/dev/null || true
rm -f /tmp/vault_test.log

echo -e "${GREEN}✅ IPC communication verified!${NC}"
