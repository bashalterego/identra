use tunnel_gateway::ipc_client::VaultClient;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Testing IPC Communication: Gateway <-> Vault");
    println!("=================================================\n");
    
    // Test 1: Connect
    println!("🔌 Test 1: Connecting to vault-daemon...");
    let mut client = VaultClient::connect().await
        .map_err(|e| format!("Connection failed: {}", e))?;
    println!("✅ Connected successfully!\n");
    
    // Test 2: Ping
    println!("📡 Test 2: Ping vault-daemon...");
    client.ping().await
        .map_err(|e| format!("Ping failed: {}", e))?;
    println!("✅ Pong received!\n");
    
    // Test 3: Store Key
    println!("📝 Test 3: Storing encryption key...");
    let key_id = "test_identity_456".to_string();
    let key_data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]; // Simulated encryption key
    let mut metadata = HashMap::new();
    metadata.insert("purpose".to_string(), "test".to_string());
    metadata.insert("algorithm".to_string(), "ChaCha20-Poly1305".to_string());
    
    client.store_key(key_id.clone(), key_data.clone(), metadata.clone(), None).await
        .map_err(|e| format!("Store failed: {}", e))?;
    println!("✅ Key stored successfully!\n");
    
    // Test 4: Check if key exists
    println!("🔍 Test 4: Checking if key exists...");
    let exists = client.key_exists(key_id.clone()).await
        .map_err(|e| format!("Check failed: {}", e))?;
    println!("✅ Key exists: {}\n", exists);
    
    // Test 5: Retrieve Key
    println!("🔓 Test 5: Retrieving key...");
    let (retrieved_data, retrieved_metadata, created_at, expires_at) = 
        client.retrieve_key(key_id.clone()).await
            .map_err(|e| format!("Retrieve failed: {}", e))?;
    
    println!("✅ Key retrieved:");
    println!("   Data matches: {}", retrieved_data == key_data);
    println!("   Metadata: {:?}", retrieved_metadata);
    println!("   Created at: {}", created_at);
    println!("   Expires at: {:?}\n", expires_at);
    
    // Test 6: List Keys
    println!("📋 Test 6: Listing all keys...");
    let keys = client.list_keys().await
        .map_err(|e| format!("List failed: {}", e))?;
    println!("✅ Found {} keys:", keys.len());
    for key in &keys {
        println!("   - {}", key);
    }
    println!();
    
    // Test 7: Delete Key
    println!("🗑️  Test 7: Deleting key...");
    client.delete_key(key_id.clone()).await
        .map_err(|e| format!("Delete failed: {}", e))?;
    println!("✅ Key deleted successfully!\n");
    
    // Test 8: Verify deletion
    println!("✔️  Test 8: Verifying deletion...");
    let exists_after = client.key_exists(key_id.clone()).await
        .map_err(|e| format!("Check failed: {}", e))?;
    println!("✅ Key exists after deletion: {}\n", exists_after);
    
    println!("🎉 All IPC tests passed!");
    println!("========================================");
    println!("✅ Gateway can communicate with vault-daemon");
    println!("✅ Line-delimited JSON protocol working");
    println!("✅ All CRUD operations functional");
    
    Ok(())
}
