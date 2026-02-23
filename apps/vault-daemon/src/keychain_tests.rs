use super::*;

#[tokio::test]
async fn test_keychain_store_retrieve_delete() {
    let storage = create_key_storage();
    
    let key_id = "test_key_integration_001";
    let test_key = b"super_secret_encryption_key_12345678";
    
    let mut custom_meta = std::collections::HashMap::new();
    custom_meta.insert("purpose".to_string(), "test".to_string());
    custom_meta.insert("algorithm".to_string(), "ChaCha20-Poly1305".to_string());
    
    let key_metadata = KeyMetadata {
        created_at: chrono::Utc::now().timestamp(),
        expires_at: None,
        custom: custom_meta,
    };
    
    // Test 1: Store key
    storage.store_key(key_id, test_key, key_metadata.clone())
        .expect("Failed to store key");
    
    // Test 2: Key should exist
    assert!(storage.key_exists(key_id), "Key should exist after storage");
    
    // Test 3: Retrieve key
    let (retrieved_key, retrieved_metadata) = storage.retrieve_key(key_id)
        .expect("Failed to retrieve key");
    
    assert_eq!(test_key, retrieved_key.as_slice(), "Retrieved key should match stored key");
    assert_eq!(key_metadata.custom.get("purpose"), retrieved_metadata.custom.get("purpose"));
    assert_eq!(key_metadata.custom.get("algorithm"), retrieved_metadata.custom.get("algorithm"));
    
    // Test 4: Delete key
    storage.delete_key(key_id)
        .expect("Failed to delete key");
    
    // Test 5: Key should not exist after deletion
    assert!(!storage.key_exists(key_id), "Key should not exist after deletion");
}

#[tokio::test]
async fn test_keychain_multiple_keys() {
    let storage = create_key_storage();
    
    let keys = vec![
        ("user_001", b"key_for_user_001_abcdefgh"),
        ("user_002", b"key_for_user_002_ijklmnop"),
        ("user_003", b"key_for_user_003_qrstuvwx"),
    ];
    
    let metadata = KeyMetadata {
        created_at: chrono::Utc::now().timestamp(),
        expires_at: None,
        custom: std::collections::HashMap::new(),
    };
    
    // Store all keys
    for (key_id, key_data) in &keys {
        storage.store_key(key_id, *key_data, metadata.clone())
            .expect(&format!("Failed to store key {}", key_id));
    }
    
    // Verify all keys exist
    for (key_id, _) in &keys {
        assert!(storage.key_exists(key_id), "Key {} should exist", key_id);
    }
    
    // Retrieve and verify all keys
    for (key_id, expected_data) in &keys {
        let (retrieved_data, _) = storage.retrieve_key(key_id)
            .expect(&format!("Failed to retrieve key {}", key_id));
        assert_eq!(expected_data.as_ref(), retrieved_data.as_slice(), "Data mismatch for key {}", key_id);
    }
    
    // Clean up - delete all keys
    for (key_id, _) in &keys {
        storage.delete_key(key_id)
            .expect(&format!("Failed to delete key {}", key_id));
        assert!(!storage.key_exists(key_id), "Key {} should be deleted", key_id);
    }
}

#[tokio::test]
async fn test_keychain_retrieve_nonexistent() {
    let storage = create_key_storage();
    
    let result = storage.retrieve_key("nonexistent_key_999");
    
    assert!(result.is_err(), "Retrieving nonexistent key should fail");
}

#[tokio::test]
async fn test_keychain_delete_nonexistent() {
    let storage = create_key_storage();
    
    let result = storage.delete_key("nonexistent_key_888");
    
    // Delete should either succeed silently or fail gracefully
    // Both behaviors are acceptable for nonexistent keys
    let _ = result;
}

#[tokio::test]
async fn test_keychain_key_overwrite() {
    let storage = create_key_storage();
    
    let key_id = "test_overwrite_key";
    let original_key = b"original_key_data_12345678";
    let new_key = b"new_key_data_87654321abcdef";
    
    let metadata = KeyMetadata {
        created_at: chrono::Utc::now().timestamp(),
        expires_at: None,
        custom: std::collections::HashMap::new(),
    };
    
    // Store original key
    storage.store_key(key_id, original_key, metadata.clone())
        .expect("Failed to store original key");
    
    // Overwrite with new key
    storage.store_key(key_id, new_key, metadata.clone())
        .expect("Failed to overwrite key");
    
    // Retrieve and verify it's the new key
    let (retrieved_key, _) = storage.retrieve_key(key_id)
        .expect("Failed to retrieve key after overwrite");
    
    assert_eq!(new_key.as_ref(), retrieved_key.as_slice(), "Retrieved key should be the new key, not the original");
    
    // Clean up
    storage.delete_key(key_id)
        .expect("Failed to delete key");
}

#[tokio::test]
async fn test_keychain_metadata_persistence() {
    let storage = create_key_storage();
    
    let key_id = "test_metadata_key";
    let key_data = b"test_key_with_metadata_123";
    
    let mut custom_metadata = std::collections::HashMap::new();
    custom_metadata.insert("user_id".to_string(), "user_12345".to_string());
    custom_metadata.insert("algorithm".to_string(), "AES-256-GCM".to_string());
    custom_metadata.insert("key_version".to_string(), "v2".to_string());
    
    let metadata = KeyMetadata {
        created_at: 1234567890,
        expires_at: Some(9999999999),
        custom: custom_metadata.clone(),
    };
    
    // Store key with metadata
    storage.store_key(key_id, key_data, metadata.clone())
        .expect("Failed to store key with metadata");
    
    // Retrieve and verify metadata
    let (_, retrieved_metadata) = storage.retrieve_key(key_id)
        .expect("Failed to retrieve key");
    
    assert_eq!(metadata.created_at, retrieved_metadata.created_at, "created_at should match");
    assert_eq!(metadata.expires_at, retrieved_metadata.expires_at, "expires_at should match");
    assert_eq!(metadata.custom.get("user_id"), retrieved_metadata.custom.get("user_id"));
    assert_eq!(metadata.custom.get("algorithm"), retrieved_metadata.custom.get("algorithm"));
    assert_eq!(metadata.custom.get("key_version"), retrieved_metadata.custom.get("key_version"));
    
    // Clean up
    storage.delete_key(key_id)
        .expect("Failed to delete key");
}
