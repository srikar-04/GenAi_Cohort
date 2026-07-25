CREATE EXTENSION IF NOT EXISTS vector;

-- Set the embedding column to a fixed size.
-- For Gemini text-embedding-004 this is typically 768 dimensions; update if your model differs.
ALTER TABLE "DocumentChunk"
  ADD COLUMN IF NOT EXISTS embedding vector(768);

CREATE INDEX IF NOT EXISTS document_chunk_embedding_idx
  ON "DocumentChunk"
  USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100);
