# Brain-Service Integration Guide
**For: Sailesh (Brain-Service Developer)**  
**From: Sarthak (Tunnel-Gateway & Vault-Daemon Owner)**  
**Date: February 22, 2026**

---

## 🎯 Architecture Clarification (CRITICAL)

**Important:** There's a key architectural distinction you need to understand:

### System Components:
1. **vault-daemon** (Port: IPC socket)
   - Purpose: Stores **encryption keys** only (not conversations)
   - Uses OS keychain (Linux Secret Service, Windows DPAPI)
   - IPC communication (not gRPC)
   - You DON'T need to call this directly

2. **tunnel-gateway** (Port: 50051) ← **THIS IS WHAT YOU NEED**
   - Purpose: Stores **conversations/memories** with embeddings
   - Uses PostgreSQL + pgvector for storage
   - gRPC communication
   - Your brain-service should connect here

### Flow for Brain-Service:
```
brain-service (Python/FastAPI)
    ↓ gRPC calls
tunnel-gateway (Rust/Tonic) on [::1]:50051
    ↓ stores in
PostgreSQL with pgvector (localhost:5432)
```

---

## 1. 📋 gRPC Specification

### Proto Files Location:
```
identra/libs/identra-proto/proto/
├── memory.proto      # ← Main API for conversations
├── auth.proto        # Authentication
├── vault.proto       # Key management (internal)
└── health.proto      # Health checks
```

### Memory Service API (Your Primary Interface):

**File:** `libs/identra-proto/proto/memory.proto`

```protobuf
service MemoryService {
  // Store a conversation/memory
  rpc StoreMemory (StoreMemoryRequest) returns (StoreMemoryResponse);
  
  // Search by semantic similarity (RAG queries)
  rpc SearchMemories (SearchMemoriesRequest) returns (SearchMemoriesResponse);
  
  // Get recent conversations (for context)
  rpc GetRecentMemories (GetRecentMemoriesRequest) returns (GetRecentMemoriesResponse);
  
  // Query with filters
  rpc QueryMemories (QueryMemoriesRequest) returns (QueryMemoriesResponse);
  
  // Get single memory by ID
  rpc GetMemory (GetMemoryRequest) returns (GetMemoryResponse);
  
  // Delete memory
  rpc DeleteMemory (DeleteMemoryRequest) returns (DeleteMemoryResponse);
}
```

---

## 2. 🔌 Conversation API Methods

### Method 1: Store Conversation
```protobuf
message StoreMemoryRequest {
  string content = 1;                    // Conversation text
  map<string, string> metadata = 2;      // user_id, session_id, model, etc.
  repeated string tags = 3;              // ["chat", "claude", "prod"]
}

message StoreMemoryResponse {
  string memory_id = 1;   // UUID of stored conversation
  bool success = 2;
  string message = 3;
}
```

**Python Example:**
```python
from identra_proto import memory_pb2

response = memory_client.StoreMemory(
    memory_pb2.StoreMemoryRequest(
        content="User: What is RAG? AI: Retrieval-Augmented Generation...",
        metadata={
            "user_id": "user_123",
            "session_id": "sess_456",
            "model": "claude-3.5-sonnet",
            "timestamp": "2026-02-22T10:30:00Z"
        },
        tags=["chat", "rag", "technical"]
    )
)
print(f"Stored with ID: {response.memory_id}")
```

### Method 2: Search (RAG Queries)
```protobuf
message SearchMemoriesRequest {
  repeated float query_embedding = 1;    // Your embedding vector
  int32 limit = 2;                       // Max results (default: 10)
  float similarity_threshold = 3;        // 0.0-1.0 (default: 0.7)
  map<string, string> filters = 4;       // user_id, tags, etc.
}

message SearchMemoriesResponse {
  repeated MemoryMatch matches = 1;
}

message MemoryMatch {
  Memory memory = 1;
  float similarity_score = 2;  // Cosine similarity
}
```

**Python Example:**
```python
# You generate embeddings in brain-service
query_embedding = your_embedding_model.encode("What did we discuss about RAG?")

response = memory_client.SearchMemories(
    memory_pb2.SearchMemoriesRequest(
        query_embedding=query_embedding.tolist(),
        limit=5,
        similarity_threshold=0.75,
        filters={"user_id": "user_123"}
    )
)

for match in response.matches:
    print(f"Score: {match.similarity_score}")
    print(f"Content: {match.memory.content}")
```

### Method 3: Get Recent Conversations
```protobuf
message GetRecentMemoriesRequest {
  int32 limit = 1;  // Number of recent conversations
}

message GetRecentMemoriesResponse {
  repeated Memory memories = 1;
}
```

**Python Example:**
```python
response = memory_client.GetRecentMemories(
    memory_pb2.GetRecentMemoriesRequest(limit=10)
)

for memory in response.memories:
    print(f"ID: {memory.id}")
    print(f"Content: {memory.content}")
    print(f"Created: {memory.created_at}")
```

---

## 3. 🔐 Authentication Flow

### Current Status:
- **Supabase Auth** integration exists but is optional
- For local development: **No authentication required**
- For production: JWT tokens via Supabase

### Local Development (No Auth):
```python
import grpc
from identra_proto import memory_pb2_grpc

# Connect without credentials
channel = grpc.insecure_channel('[::1]:50051')
memory_client = memory_pb2_grpc.MemoryServiceStub(channel)
```

### Production (With Auth):
```python
# When auth is enabled, you'll need:
class AuthInterceptor(grpc.UnaryUnaryClientInterceptor):
    def __init__(self, token):
        self.token = token
    
    def intercept_unary_unary(self, continuation, client_call_details, request):
        metadata = [('authorization', f'Bearer {self.token}')]
        new_details = client_call_details._replace(metadata=metadata)
        return continuation(new_details, request)

channel = grpc.insecure_channel('[::1]:50051')
intercepted_channel = grpc.intercept_channel(channel, AuthInterceptor(jwt_token))
memory_client = memory_pb2_grpc.MemoryServiceStub(intercepted_channel)
```

---

## 4. 📊 Data Format

### Memory Object Structure:
```protobuf
message Memory {
  string id = 1;                              // UUID
  string content = 2;                         // Conversation text
  map<string, string> metadata = 3;           // Flexible key-value pairs
  repeated float embedding = 4;               // 384-dim vector (not returned in searches)
  google.protobuf.Timestamp created_at = 5;   // ISO timestamp
  google.protobuf.Timestamp updated_at = 6;   // ISO timestamp
  repeated string tags = 7;                   // ["chat", "prod", "user_123"]
}
```

### Metadata Schema (Recommended):
```python
metadata = {
    "user_id": str,           # User identifier
    "session_id": str,        # Conversation session
    "model": str,             # AI model used (claude-3.5-sonnet, gpt-4, etc.)
    "type": str,              # "user_message", "ai_response", "system"
    "timestamp": str,         # ISO 8601 format
    "token_count": str,       # Optional: number of tokens
    "parent_id": str          # Optional: reference to previous message
}
```

### Constraints:
- **content**: Max 10MB (practical limit)
- **tags**: Max 50 tags per memory
- **metadata**: Max 100 key-value pairs
- **embedding**: 384 dimensions (AllMiniLML6V2 model)

---

## 5. 🌐 Connection Details

### Tunnel-Gateway:
- **Host**: `[::1]` (IPv6 localhost) or `127.0.0.1`
- **Port**: `50051`
- **Protocol**: gRPC (HTTP/2)
- **TLS**: Not enabled for local dev (use `insecure_channel`)

### Health Check Endpoint:
```python
from identra_proto import health_pb2, health_pb2_grpc

health_client = health_pb2_grpc.HealthStub(channel)
response = health_client.Check(health_pb2.HealthCheckRequest())
print(f"Status: {response.status}")  # SERVING, NOT_SERVING, UNKNOWN
```

### Connection Pooling (Recommended):
```python
import grpc

# Reuse channels (they maintain connection pools internally)
channel = grpc.insecure_channel(
    '[::1]:50051',
    options=[
        ('grpc.keepalive_time_ms', 10000),
        ('grpc.keepalive_timeout_ms', 5000),
        ('grpc.keepalive_permit_without_calls', True),
        ('grpc.http2.max_pings_without_data', 0),
    ]
)
```

---

## 6. ⚠️ Error Handling

### gRPC Status Codes:
```python
import grpc

try:
    response = memory_client.StoreMemory(request)
except grpc.RpcError as e:
    status_code = e.code()
    details = e.details()
    
    if status_code == grpc.StatusCode.UNAVAILABLE:
        # Gateway is down - retry with backoff
        print("Gateway unavailable, retrying...")
    
    elif status_code == grpc.StatusCode.INVALID_ARGUMENT:
        # Bad request - check your data
        print(f"Invalid argument: {details}")
    
    elif status_code == grpc.StatusCode.DEADLINE_EXCEEDED:
        # Timeout - increase deadline
        print("Request timeout")
    
    elif status_code == grpc.StatusCode.INTERNAL:
        # Server error - log and alert
        print(f"Internal error: {details}")
```

### Retry Strategy (Recommended):
```python
import time
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10)
)
def store_memory_with_retry(client, request):
    return client.StoreMemory(request)
```

### Offline Fallback:
```python
class MemoryClient:
    def __init__(self):
        self.channel = grpc.insecure_channel('[::1]:50051')
        self.stub = memory_pb2_grpc.MemoryServiceStub(self.channel)
        self.local_queue = []  # SQLite fallback
    
    def store_memory(self, content, metadata, tags):
        try:
            # Try gateway first
            return self.stub.StoreMemory(...)
        except grpc.RpcError:
            # Fallback to local SQLite
            self.local_queue.append({
                'content': content,
                'metadata': metadata,
                'tags': tags
            })
            return {'memory_id': 'local_' + str(uuid.uuid4())}
    
    def sync_queue(self):
        # Periodically sync queued items when gateway is available
        for item in self.local_queue:
            try:
                self.stub.StoreMemory(...)
                self.local_queue.remove(item)
            except:
                break
```

---

## 7. 🚀 Setup Instructions

### Step 1: Install Prerequisites
```bash
# On your development machine
sudo apt-get install -y protobuf-compiler

# Python dependencies
pip install grpcio grpcio-tools
```

### Step 2: Generate Python Stubs
```bash
cd /home/starkalterego/projects/identra

# Generate Python client code from proto files
python -m grpc_tools.protoc \
    -I libs/identra-proto/proto \
    --python_out=../brain-service/generated \
    --grpc_python_out=../brain-service/generated \
    libs/identra-proto/proto/memory.proto \
    libs/identra-proto/proto/health.proto

# This creates:
# - memory_pb2.py
# - memory_pb2_grpc.py
# - health_pb2.py
# - health_pb2_grpc.py
```

### Step 3: Start Tunnel-Gateway Locally
```bash
cd /home/starkalterego/projects/identra

# Terminal 1: Start database
docker start identra-db || docker run --name identra-db \
    -e POSTGRES_PASSWORD=password \
    -d -p 5432:5432 \
    pgvector/pgvector:pg16

# Terminal 2: Start gateway
just dev-gateway

# You should see:
# Starting Gateway...
# Listening on [::1]:50051
```

### Step 4: Test Connection from Python
```python
# test_gateway.py
import grpc
from generated import memory_pb2, memory_pb2_grpc

def test_connection():
    channel = grpc.insecure_channel('[::1]:50051')
    client = memory_pb2_grpc.MemoryServiceStub(channel)
    
    # Store a test memory
    response = client.StoreMemory(
        memory_pb2.StoreMemoryRequest(
            content="Test conversation",
            metadata={"test": "true"},
            tags=["test"]
        )
    )
    
    print(f"✅ Connection successful!")
    print(f"Memory ID: {response.memory_id}")

if __name__ == "__main__":
    test_connection()
```

### Step 5: Run the Test
```bash
python test_gateway.py
```

---

## 📚 Additional Resources

### Proto Files:
- **Memory API**: `identra/libs/identra-proto/proto/memory.proto`
- **Auth API**: `identra/libs/identra-proto/proto/auth.proto`
- **Health API**: `identra/libs/identra-proto/proto/health.proto`

### Database Schema:
The gateway stores memories in PostgreSQL with this schema:
```sql
CREATE TABLE memories (
    id UUID PRIMARY KEY,
    content TEXT NOT NULL,
    embedding VECTOR(384),  -- pgvector extension
    metadata JSONB,
    tags TEXT[],
    created_at BIGINT,
    updated_at BIGINT
);

CREATE INDEX ON memories USING ivfflat (embedding vector_cosine_ops);
```

### Embedding Model:
- **Model**: AllMiniLML6V2 (sentence-transformers)
- **Dimensions**: 384
- **Gateway handles embeddings**: You DON'T need to generate them (just pass content)
- **For RAG**: You can generate embeddings in brain-service and pass to SearchMemories

---

## 🔄 Migration from SQLite

### Current Brain-Service (SQLite):
```python
# Old code
cursor.execute("INSERT INTO conversations (content) VALUES (?)", (content,))
```

### New Brain-Service (Gateway):
```python
# New code
response = memory_client.StoreMemory(
    memory_pb2.StoreMemoryRequest(
        content=content,
        metadata={"migrated_from": "sqlite"},
        tags=["conversation"]
    )
)
```

### Migration Script Example:
```python
import sqlite3
import grpc
from generated import memory_pb2, memory_pb2_grpc

def migrate_sqlite_to_gateway():
    # Connect to both
    sqlite_conn = sqlite3.connect('conversations.db')
    grpc_channel = grpc.insecure_channel('[::1]:50051')
    memory_client = memory_pb2_grpc.MemoryServiceStub(grpc_channel)
    
    # Migrate
    cursor = sqlite_conn.execute("SELECT id, content, timestamp FROM conversations")
    for row in cursor:
        response = memory_client.StoreMemory(
            memory_pb2.StoreMemoryRequest(
                content=row[1],
                metadata={
                    "sqlite_id": str(row[0]),
                    "migrated_at": row[2]
                },
                tags=["migrated"]
            )
        )
        print(f"Migrated: {row[0]} -> {response.memory_id}")
```

---

## 🐛 Troubleshooting

### Issue: "failed to connect to all addresses"
**Solution**: Ensure tunnel-gateway is running on port 50051
```bash
just dev-gateway
# OR
cargo run --bin tunnel-gateway
```

### Issue: "Database connection failed"
**Solution**: Start PostgreSQL container
```bash
docker start identra-db
```

### Issue: "Module 'memory_pb2' not found"
**Solution**: Regenerate Python stubs
```bash
python -m grpc_tools.protoc -I libs/identra-proto/proto \
    --python_out=. --grpc_python_out=. \
    libs/identra-proto/proto/memory.proto
```

### Issue: "Embedding generation slow"
**Solution**: Gateway uses CPU-based embeddings. Consider:
- Batching requests
- Caching embeddings
- Generating embeddings in brain-service (more control)

---

## 📞 Contact

**Sarthak** (Tunnel-Gateway & Vault-Daemon Owner)
- Slack: @sarthak
- Email: sarthak@identra.dev
- Issues: https://github.com/identra/identra/issues

**For urgent issues**: Ping me on Slack with `@sarthak [URGENT]`

---

## ✅ Checklist for Phase 3

- [ ] Generate Python stubs from proto files
- [ ] Test connection to tunnel-gateway on port 50051
- [ ] Replace SQLite `INSERT` with `StoreMemory` gRPC calls
- [ ] Replace SQLite `SELECT` with `SearchMemories` for RAG
- [ ] Implement retry logic with exponential backoff
- [ ] Add offline fallback (optional)
- [ ] Test end-to-end conversation flow
- [ ] Benchmark latency (should be <100ms for storage, <200ms for search)
- [ ] Run migration script from SQLite to gateway

**Timeline**: Feb 9-12 ✅ (You're ahead of schedule!)

Good luck with Phase 3! Let me know if you need any clarification.

- Sarthak
