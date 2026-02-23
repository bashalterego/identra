#!/usr/bin/env python3
"""
Example client for Identra Tunnel-Gateway Memory Service
For: Sailesh (Brain-Service Developer)
"""

import grpc
from typing import List, Dict, Optional
import sys
import os

# Add generated proto files to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'generated'))

try:
    from generated import memory_pb2, memory_pb2_grpc
except ImportError:
    print("❌ Error: Proto files not generated!")
    print("Run: python -m grpc_tools.protoc -I libs/identra-proto/proto --python_out=. --grpc_python_out=. libs/identra-proto/proto/memory.proto")
    sys.exit(1)


class MemoryClient:
    """Python client for Identra Memory Service"""
    
    def __init__(self, host='[::1]', port=50051):
        """
        Initialize connection to tunnel-gateway
        
        Args:
            host: Gateway host (default: [::1] for IPv6 localhost)
            port: Gateway port (default: 50051)
        """
        self.channel = grpc.insecure_channel(f'{host}:{port}')
        self.stub = memory_pb2_grpc.MemoryServiceStub(self.channel)
    
    def store_conversation(
        self, 
        content: str, 
        metadata: Optional[Dict[str, str]] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Store a conversation in encrypted vault
        
        Args:
            content: Conversation text
            metadata: Optional metadata (user_id, session_id, model, etc.)
            tags: Optional tags for categorization
            
        Returns:
            memory_id: UUID of stored conversation
        """
        request = memory_pb2.StoreMemoryRequest(
            content=content,
            metadata=metadata or {},
            tags=tags or []
        )
        
        try:
            response = self.stub.StoreMemory(request)
            if response.success:
                print(f"✅ Stored conversation: {response.memory_id}")
                return response.memory_id
            else:
                print(f"❌ Storage failed: {response.message}")
                return None
        except grpc.RpcError as e:
            print(f"❌ gRPC Error: {e.code()} - {e.details()}")
            return None
    
    def search_conversations(
        self,
        query_embedding: List[float],
        limit: int = 10,
        similarity_threshold: float = 0.7,
        filters: Optional[Dict[str, str]] = None
    ) -> List[Dict]:
        """
        Search conversations by semantic similarity (RAG)
        
        Args:
            query_embedding: Embedding vector for query
            limit: Max number of results
            similarity_threshold: Min similarity score (0.0-1.0)
            filters: Optional filters (user_id, tags, etc.)
            
        Returns:
            List of matching conversations with scores
        """
        request = memory_pb2.SearchMemoriesRequest(
            query_embedding=query_embedding,
            limit=limit,
            similarity_threshold=similarity_threshold,
            filters=filters or {}
        )
        
        try:
            response = self.stub.SearchMemories(request)
            results = []
            for match in response.matches:
                results.append({
                    'id': match.memory.id,
                    'content': match.memory.content,
                    'similarity': match.similarity_score,
                    'metadata': dict(match.memory.metadata),
                    'tags': list(match.memory.tags),
                    'created_at': match.memory.created_at.ToDatetime()
                })
            print(f"✅ Found {len(results)} matches")
            return results
        except grpc.RpcError as e:
            print(f"❌ gRPC Error: {e.code()} - {e.details()}")
            return []
    
    def get_recent_conversations(self, limit: int = 10) -> List[Dict]:
        """
        Get recent conversations (for context)
        
        Args:
            limit: Number of recent conversations
            
        Returns:
            List of recent conversations
        """
        request = memory_pb2.GetRecentMemoriesRequest(limit=limit)
        
        try:
            response = self.stub.GetRecentMemories(request)
            results = []
            for memory in response.memories:
                results.append({
                    'id': memory.id,
                    'content': memory.content,
                    'metadata': dict(memory.metadata),
                    'tags': list(memory.tags),
                    'created_at': memory.created_at.ToDatetime()
                })
            print(f"✅ Retrieved {len(results)} recent conversations")
            return results
        except grpc.RpcError as e:
            print(f"❌ gRPC Error: {e.code()} - {e.details()}")
            return []
    
    def close(self):
        """Close the gRPC channel"""
        self.channel.close()


# Example usage
if __name__ == "__main__":
    print("🔌 Connecting to Identra Gateway...")
    
    client = MemoryClient()
    
    # Example 1: Store a conversation
    print("\n📝 Example 1: Store Conversation")
    memory_id = client.store_conversation(
        content="User: What is RAG?\nAI: RAG stands for Retrieval-Augmented Generation...",
        metadata={
            "user_id": "user_123",
            "session_id": "sess_456",
            "model": "claude-3.5-sonnet",
            "type": "chat"
        },
        tags=["chat", "rag", "technical"]
    )
    
    # Example 2: Search conversations (you need to generate embeddings)
    print("\n🔍 Example 2: Search Conversations")
    # NOTE: You need to generate embeddings in your brain-service
    # This is a placeholder - replace with your actual embedding
    dummy_embedding = [0.1] * 384  # 384-dimensional vector
    
    matches = client.search_conversations(
        query_embedding=dummy_embedding,
        limit=5,
        similarity_threshold=0.7,
        filters={"user_id": "user_123"}
    )
    
    for i, match in enumerate(matches, 1):
        print(f"\nMatch {i}:")
        print(f"  Score: {match['similarity']:.3f}")
        print(f"  Content: {match['content'][:100]}...")
        print(f"  Tags: {match['tags']}")
    
    # Example 3: Get recent conversations
    print("\n📚 Example 3: Get Recent Conversations")
    recent = client.get_recent_conversations(limit=5)
    
    for i, conv in enumerate(recent, 1):
        print(f"\nConversation {i}:")
        print(f"  ID: {conv['id']}")
        print(f"  Content: {conv['content'][:100]}...")
        print(f"  Created: {conv['created_at']}")
    
    client.close()
    print("\n✅ Done!")
