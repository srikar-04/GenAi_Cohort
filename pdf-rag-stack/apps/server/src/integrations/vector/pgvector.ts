import { Prisma } from '@prisma/client';
import { prisma } from '../../db/prisma';
import { Chunk } from '../chunking/chunker';

const toVector = (embedding: number[]) => `[${embedding.join(',')}]`;

export const upsertChunks = async (
  documentId: string,
  chunks: Chunk[],
  embeddings: number[][],
) => {
  if (chunks.length !== embeddings.length) {
    throw new Error('Chunk/embedding length mismatch');
  }

  for (let i = 0; i < chunks.length; i += 1) {
    const chunk = chunks[i];
    const vector = toVector(embeddings[i]);
    const metadata = chunk.metadata ? JSON.stringify(chunk.metadata) : null;

    await prisma.$executeRaw(
      Prisma.sql`
        INSERT INTO "DocumentChunk" ("documentId", "chunkIndex", "content", "tokenCount", "metadata", "embedding")
        VALUES (${documentId}, ${i}, ${chunk.content}, ${chunk.tokenCount}, ${metadata}::jsonb, ${Prisma.raw(`'${vector}'::vector`)})
        ON CONFLICT ("documentId", "chunkIndex")
        DO UPDATE SET
          "content" = EXCLUDED."content",
          "tokenCount" = EXCLUDED."tokenCount",
          "metadata" = EXCLUDED."metadata",
          "embedding" = EXCLUDED."embedding"
      `,
    );
  }
};

export const similaritySearch = async (params: {
  userId: string;
  documentId?: string;
  embedding: number[];
  topK: number;
  minScore: number;
}) => {
  const { userId, documentId, embedding, topK, minScore } = params;
  const vector = toVector(embedding);

  const rows = await prisma.$queryRaw<
    {
      id: string;
      documentId: string;
      content: string;
      metadata: Prisma.JsonValue;
      score: number;
    }[]
  >(
    Prisma.sql`
      SELECT
        c."id",
        c."documentId",
        c."content",
        c."metadata",
        1 - (c."embedding" <=> ${Prisma.raw(`'${vector}'::vector`)}) as score
      FROM "DocumentChunk" c
      JOIN "Document" d ON d."id" = c."documentId"
      WHERE d."userId" = ${userId}
      ${documentId ? Prisma.sql`AND c."documentId" = ${documentId}` : Prisma.empty}
      ORDER BY "embedding" <=> ${Prisma.raw(`'${vector}'::vector`)}
      LIMIT ${topK}
    `,
  );

  return rows.filter((row) => row.score >= minScore);
};
