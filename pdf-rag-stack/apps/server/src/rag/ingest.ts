import { DocumentStatus, DocumentType } from '@prisma/client';
import { prisma } from '../db/prisma';
import { cloudinaryStorage } from '../integrations/storage/cloudinary';
import { loadPdfText } from '../integrations/pdf/pdf-loader';
import { chunkText } from '../integrations/chunking/chunker';
import { chunkResearchPaper } from '../integrations/chunking/research-chunker';
import { embedTexts } from '../integrations/embeddings/gemini-embeddings';
import { upsertChunks } from '../integrations/vector/pgvector';
import { normalizeWhitespace } from '../common/utils/text';

const DEFAULT_CHUNK_SIZE = 450;
const DEFAULT_OVERLAP = 80;

export const ingestDocument = async (documentId: string) => {
  const document = await prisma.document.findUnique({ where: { id: documentId } });
  if (!document) {
    throw new Error('Document not found');
  }

  await prisma.document.update({
    where: { id: documentId },
    data: { status: DocumentStatus.PROCESSING },
  });

  await prisma.documentIngestion.upsert({
    where: { documentId },
    update: { startedAt: new Date(), error: null },
    create: { documentId, startedAt: new Date() },
  });

  try {
    const buffer = await cloudinaryStorage.getObjectBuffer({
      url: document.storageUrl,
      publicId: document.storageKey,
    });
    const { pages, pageCount } = await loadPdfText(buffer);
    const rawText = normalizeWhitespace(pages.join('\n\n'));
    const wordCount = rawText.split(/\s+/).filter(Boolean).length;

    const chunks =
      document.documentType === DocumentType.RESEARCH
        ? chunkResearchPaper(pages, {
            chunkSize: DEFAULT_CHUNK_SIZE,
            overlap: DEFAULT_OVERLAP,
          })
        : chunkText(rawText, {
            chunkSize: DEFAULT_CHUNK_SIZE,
            overlap: DEFAULT_OVERLAP,
          });

    await prisma.documentChunk.deleteMany({ where: { documentId } });

    const embeddings = await embedTexts(chunks.map((chunk) => chunk.content));
    await upsertChunks(documentId, chunks, embeddings);

    await prisma.document.update({
      where: { id: documentId },
      data: {
        status: DocumentStatus.READY,
        pageCount,
        wordCount,
      },
    });

    await prisma.documentIngestion.update({
      where: { documentId },
      data: {
        finishedAt: new Date(),
        chunkCount: chunks.length,
        embeddingDim: embeddings[0]?.length ?? null,
        model: process.env.GEMINI_EMBEDDING_MODEL,
      },
    });
  } catch (error) {
    await prisma.document.update({
      where: { id: documentId },
      data: { status: DocumentStatus.FAILED },
    });
    await prisma.documentIngestion.update({
      where: { documentId },
      data: { finishedAt: new Date(), error: (error as Error).message },
    });
    throw error;
  }
};
