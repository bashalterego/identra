use tonic::transport::Server;
use std::sync::Arc;
use dotenvy::dotenv;
use std::env;

mod database;
mod services;
pub mod ipc_client;
mod auth;

use database::MemoryDatabase;
use services::memory::MemoryServiceImpl;
use services::vault::VaultServiceImpl;
use auth::{SupabaseClient, AuthServiceImpl};
use identra_proto::auth::auth_service_server::AuthServiceServer;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load .env variables
    dotenv().ok(); 
    tracing_subscriber::fmt::init();

    tracing::info!("Starting Gateway...");

    // Connect to Postgres
    let db_url = env::var("DATABASE_URL").expect("DATABASE_URL must be set in .env");
    let db = Arc::new(MemoryDatabase::connect(&db_url).await?);

    // Initialize Supabase client for authentication
    let supabase = Arc::new(SupabaseClient::new()?);
    tracing::info!("Supabase Auth client initialized");

    // Initialize services
    let memory_service = MemoryServiceImpl::new(db.clone());
    let auth_service = AuthServiceImpl::new(supabase);
    let vault_service = VaultServiceImpl::new();

    let addr = "[::1]:50051".parse()?;
    tracing::info!("Listening on {}", addr);

    Server::builder()
        .add_service(memory_service.into_server())
        .add_service(AuthServiceServer::new(auth_service))
        .add_service(vault_service.into_server())
        .serve(addr)
        .await?;

    Ok(())
}