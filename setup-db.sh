#!/bin/bash
set -e

echo "🔍 Checking if database container is running..."
if ! docker ps | grep -q identra-db; then
    echo "❌ Database container not running"
    echo "Starting database..."
    docker run --name identra-db -e POSTGRES_PASSWORD=password -d -p 5432:5432 pgvector/pgvector:pg16 2>/dev/null || \
    docker start identra-db
    sleep 3
fi

echo "✅ Database container is running"
echo ""
echo "📊 Installing pgvector extension..."
docker exec identra-db psql -U postgres -c "CREATE EXTENSION IF NOT EXISTS vector;" 2>/dev/null || true

echo ""
echo "🗄️  Database ready!"
echo "   Connection: postgresql://postgres:password@localhost:5432/postgres"
echo ""
echo "🚀 You can now run:"
echo "   just dev-gateway  (in one terminal)"
echo "   just dev-vault    (in another terminal)"
