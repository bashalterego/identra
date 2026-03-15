use interprocess::local_socket::{
    tokio::{prelude::*, Stream},
    GenericNamespaced,
};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

#[cfg(windows)]
const IPC_PIPE_NAME: &str = "@identra-vault";

#[cfg(unix)]
const IPC_PIPE_NAME: &str = "/tmp/identra-vault.sock";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VaultRequest {
    StoreKey { 
        key_id: String, 
        key_data: Vec<u8>,
        metadata: std::collections::HashMap<String, String>,
        expires_at: Option<i64>,
    },
    RetrieveKey { key_id: String },
    DeleteKey { key_id: String },
    KeyExists { key_id: String },
    ListKeys,
    Ping,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VaultResponse {
    Success,
    KeyData { 
        key_data: Vec<u8>,
        metadata: std::collections::HashMap<String, String>,
        created_at: i64,
        expires_at: Option<i64>,
    },
    KeyList(Vec<String>),
    Exists(bool),
    Error(String),
    Pong,
}

#[derive(Debug)]
pub enum VaultClientError {
    ConnectionFailed(String),
    SendFailed(String),
    ReceiveFailed(String),
    SerializationError(String),
}

impl fmt::Display for VaultClientError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ConnectionFailed(msg) => write!(f, "Failed to connect to vault: {}", msg),
            Self::SendFailed(msg) => write!(f, "Failed to send request: {}", msg),
            Self::ReceiveFailed(msg) => write!(f, "Failed to receive response: {}", msg),
            Self::SerializationError(msg) => write!(f, "Serialization error: {}", msg),
        }
    }
}

impl Error for VaultClientError {}

pub struct VaultClient {
    reader: BufReader<tokio::io::ReadHalf<Stream>>,
    writer: tokio::io::WriteHalf<Stream>,
}

impl VaultClient {
    pub async fn connect() -> Result<Self, VaultClientError> {
        let name = IPC_PIPE_NAME.to_ns_name::<GenericNamespaced>()
            .map_err(|e| VaultClientError::ConnectionFailed(e.to_string()))?;
        
        let stream = Stream::connect(name)
            .await
            .map_err(|e| VaultClientError::ConnectionFailed(e.to_string()))?;

        let (reader, writer) = tokio::io::split(stream);
        let reader = BufReader::new(reader);

        Ok(Self { reader, writer })
    }

    pub async fn send_request(&mut self, request: VaultRequest) -> Result<VaultResponse, VaultClientError> {
        // Serialize request to JSON
        let request_json = serde_json::to_string(&request)
            .map_err(|e| VaultClientError::SerializationError(e.to_string()))?;
        
        // Send line-delimited JSON (matches vault-daemon protocol)
        self.writer.write_all(request_json.as_bytes())
            .await
            .map_err(|e| VaultClientError::SendFailed(e.to_string()))?;
        
        self.writer.write_all(b"\n")
            .await
            .map_err(|e| VaultClientError::SendFailed(e.to_string()))?;
        
        self.writer.flush()
            .await
            .map_err(|e| VaultClientError::SendFailed(e.to_string()))?;

        // Read response line
        let mut response_line = String::new();
        self.reader.read_line(&mut response_line)
            .await
            .map_err(|e| VaultClientError::ReceiveFailed(e.to_string()))?;

        // Deserialize response
        let response: VaultResponse = serde_json::from_str(&response_line)
            .map_err(|e| VaultClientError::SerializationError(e.to_string()))?;

        Ok(response)
    }

    pub async fn store_key(
        &mut self, 
        key_id: String, 
        key_data: Vec<u8>,
        metadata: std::collections::HashMap<String, String>,
        expires_at: Option<i64>,
    ) -> Result<(), VaultClientError> {
        let response = self.send_request(VaultRequest::StoreKey { key_id, key_data, metadata, expires_at }).await?;
        match response {
            VaultResponse::Success => Ok(()),
            VaultResponse::Error(message) => Err(VaultClientError::ReceiveFailed(message)),
            _ => Err(VaultClientError::ReceiveFailed("Unexpected response type".to_string())),
        }
    }

    pub async fn retrieve_key(&mut self, key_id: String) -> Result<(Vec<u8>, std::collections::HashMap<String, String>, i64, Option<i64>), VaultClientError> {
        let response = self.send_request(VaultRequest::RetrieveKey { key_id }).await?;
        match response {
            VaultResponse::KeyData { key_data, metadata, created_at, expires_at } => {
                Ok((key_data, metadata, created_at, expires_at))
            }
            VaultResponse::Error(message) => Err(VaultClientError::ReceiveFailed(message)),
            _ => Err(VaultClientError::ReceiveFailed("Unexpected response type".to_string())),
        }
    }

    pub async fn delete_key(&mut self, key_id: String) -> Result<(), VaultClientError> {
        let response = self.send_request(VaultRequest::DeleteKey { key_id }).await?;
        match response {
            VaultResponse::Success => Ok(()),
            VaultResponse::Error(message) => Err(VaultClientError::ReceiveFailed(message)),
            _ => Err(VaultClientError::ReceiveFailed("Unexpected response type".to_string())),
        }
    }

    pub async fn key_exists(&mut self, key_id: String) -> Result<bool, VaultClientError> {
        let response = self.send_request(VaultRequest::KeyExists { key_id }).await?;
        match response {
            VaultResponse::Exists(exists) => Ok(exists),
            VaultResponse::Error(message) => Err(VaultClientError::ReceiveFailed(message)),
            _ => Err(VaultClientError::ReceiveFailed("Unexpected response type".to_string())),
        }
    }
    
    pub async fn list_keys(&mut self) -> Result<Vec<String>, VaultClientError> {
        let response = self.send_request(VaultRequest::ListKeys).await?;
        match response {
            VaultResponse::KeyList(keys) => Ok(keys),
            VaultResponse::Error(message) => Err(VaultClientError::ReceiveFailed(message)),
            _ => Err(VaultClientError::ReceiveFailed("Unexpected response type".to_string())),
        }
    }
    
    pub async fn ping(&mut self) -> Result<(), VaultClientError> {
        let response = self.send_request(VaultRequest::Ping).await?;
        match response {
            VaultResponse::Pong => Ok(()),
            VaultResponse::Error(message) => Err(VaultClientError::ReceiveFailed(message)),
            _ => Err(VaultClientError::ReceiveFailed("Unexpected response type".to_string())),
        }
    }
}
