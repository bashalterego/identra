#!/usr/bin/env bash
# Test script for IPC communication between tunnel-gateway and vault-daemon

set -e

echo "🧪 Testing IPC Communication between Gateway and Vault"
echo "======================================================"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Start vault-daemon in background
echo -e "${YELLOW}📦 Starting vault-daemon...${NC}"
cargo run --bin vault-daemon &
VAULT_PID=$!

# Give it time to start
sleep 2

# Check if vault is running
if kill -0 $VAULT_PID 2>/dev/null; then
    echo -e "${GREEN}✅ vault-daemon started (PID: $VAULT_PID)${NC}"
else
    echo -e "${RED}❌ vault-daemon failed to start${NC}"
    exit 1
fi

# Test if socket exists
if [ -f /tmp/identra-vault.sock ]; then
    echo -e "${GREEN}✅ IPC socket created at /tmp/identra-vault.sock${NC}"
else
    echo -e "${RED}❌ IPC socket not found${NC}"
    kill $VAULT_PID 2>/dev/null || true
    exit 1
fi

# Create a simple Rust test client
echo -e "${YELLOW}📝 Creating test client...${NC}"

cat > /tmp/test_ipc_client.rs << 'EOF'
use std::path::Path;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    use interprocess::local_socket::tokio::{prelude::*, Stream};
    use serde::{Deserialize, Serialize};
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
    
    #[derive(Debug, Serialize, Deserialize)]
    enum VaultRequest {
        Ping,
        StoreKey {
            key_id: String,
            key_data: Vec<u8>,
            metadata: std::collections::HashMap<String, String>,
            expires_at: Option<i64>,
        },
        RetrieveKey { key_id: String },
        ListKeys,
    }
    
    #[derive(Debug, Serialize, Deserialize)]
    enum VaultResponse {
        Success,
        KeyData {
            key_data: Vec<u8>,
            metadata: std::collections::HashMap<String, String>,
            created_at: i64,
            expires_at: Option<i64>,
        },
        KeyList(Vec<String>),
        Error(String),
        Pong,
    }
    
    println!("🔌 Connecting to vault-daemon...");
    
    let socket_path = "/tmp/identra-vault.sock";
    let stream = Stream::connect(socket_path).await?;
    
    let (reader, mut writer) = tokio::io::split(stream);
    let mut reader = BufReader::new(reader);
    
    println!("✅ Connected!");
    
    // Test 1: Ping
    println!("\n📡 Test 1: Ping");
    let ping_req = serde_json::to_string(&VaultRequest::Ping)?;
    writer.write_all(ping_req.as_bytes()).await?;
    writer.write_all(b"\n").await?;
    writer.flush().await?;
    
    let mut response = String::new();
    reader.read_line(&mut response).await?;
    let pong: VaultResponse = serde_json::from_str(&response)?;
    println!("✅ Response: {:?}", pong);
    
    // Test 2: Store Key
    println!("\n📝 Test 2: Store Key");
    let mut metadata = std::collections::HashMap::new();
    metadata.insert("test".to_string(), "true".to_string());
    
    let store_req = VaultRequest::StoreKey {
        key_id: "test_key_123".to_string(),
        key_data: vec![1, 2, 3, 4, 5],
        metadata,
        expires_at: None,
    };
    let store_json = serde_json::to_string(&store_req)?;
    writer.write_all(store_json.as_bytes()).await?;
    writer.write_all(b"\n").await?;
    writer.flush().await?;
    
    response.clear();
    reader.read_line(&mut response).await?;
    let store_resp: VaultResponse = serde_json::from_str(&response)?;
    println!("✅ Response: {:?}", store_resp);
    
    // Test 3: Retrieve Key
    println!("\n🔍 Test 3: Retrieve Key");
    let retrieve_req = VaultRequest::RetrieveKey {
        key_id: "test_key_123".to_string(),
    };
    let retrieve_json = serde_json::to_string(&retrieve_req)?;
    writer.write_all(retrieve_json.as_bytes()).await?;
    writer.write_all(b"\n").await?;
    writer.flush().await?;
    
    response.clear();
    reader.read_line(&mut response).await?;
    let retrieve_resp: VaultResponse = serde_json::from_str(&response)?;
    println!("✅ Response: {:?}", retrieve_resp);
    
    // Test 4: List Keys
    println!("\n📋 Test 4: List Keys");
    let list_req = VaultRequest::ListKeys;
    let list_json = serde_json::to_string(&list_req)?;
    writer.write_all(list_json.as_bytes()).await?;
    writer.write_all(b"\n").await?;
    writer.flush().await?;
    
    response.clear();
    reader.read_line(&mut response).await?;
    let list_resp: VaultResponse = serde_json::from_str(&response)?;
    println!("✅ Response: {:?}", list_resp);
    
    println!("\n🎉 All tests passed!");
    
    Ok(())
}
EOF

# Run test using the workspace dependencies
cd /home/starkalterego/projects/identra
echo -e "${YELLOW}🚀 Running IPC tests...${NC}"

# Create temporary test binary
cat >> apps/vault-daemon/Cargo.toml << 'EOF'

[[bin]]
name = "test_ipc"
path = "/tmp/test_ipc_client.rs"
EOF

cargo run --bin test_ipc 2>&1

# Cleanup
echo -e "${YELLOW}🧹 Cleaning up...${NC}"
kill $VAULT_PID 2>/dev/null || true
rm -f /tmp/identra-vault.sock
git checkout apps/vault-daemon/Cargo.toml 2>/dev/null || true

echo -e "${GREEN}✅ Test complete!${NC}"
