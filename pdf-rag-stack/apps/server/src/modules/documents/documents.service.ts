import { Document, DocumentStatus, DocumentType } from '@prisma/client';
import { prisma } from '../../db/prisma';
import { cloudinaryStorage } from '../../integrations/storage/cloudinary';
import { sha256 } from '../../common/utils/hash';
import { ingestDocument } from '../../rag/ingest';

const sanitizeFileName = (fileName: string) =>
  fileName.replace(/[^a-zA-Z0-9._-]/g, '_').slice(0, 120);

export const documentsService = {
  async createFromUpload(params: {
    userId: string;
    fileName: string;
    mimeType: string;
    fileSize: number;
    buffer: Buffer;
    documentType?: DocumentType;
    title?: string;
  }) {
    const safeName = sanitizeFileName(params.fileName);
    const checksum = sha256(params.buffer);

    const uploadResult = await cloudinaryStorage.uploadBuffer({
      buffer: params.buffer,
      fileName: params.fileName,
      folder: params.userId,
    });

    const document = await prisma.document.create({
      data: {
        userId: params.userId,
        title: params.title ?? safeName,
        fileName: params.fileName,
        fileSize: params.fileSize,
        mimeType: params.mimeType,
        storageKey: uploadResult.key,
        storageUrl: uploadResult.url,
        checksum,
        status: DocumentStatus.UPLOADED,
        documentType: params.documentType ?? DocumentType.GENERAL,
      },
    });

    void ingestDocument(document.id);
    return document;
  },

  async registerPresignedUpload(params: {
    userId: string;
    fileName: string;
    mimeType: string;
    fileSize: number;
    storageKey: string;
    storageUrl: string;
    checksum: string;
    documentType?: DocumentType;
  }) {
    const document = await prisma.document.create({
      data: {
        userId: params.userId,
        title: params.fileName,
        fileName: params.fileName,
        fileSize: params.fileSize,
        mimeType: params.mimeType,
        storageKey: params.storageKey,
        storageUrl: params.storageUrl,
        checksum: params.checksum,
        status: DocumentStatus.UPLOADED,
        documentType: params.documentType ?? DocumentType.GENERAL,
      },
    });

    void ingestDocument(document.id);
    return document;
  },

  async listDocuments(userId: string, status?: DocumentStatus) {
    return prisma.document.findMany({
      where: { userId, status },
      orderBy: { createdAt: 'desc' },
    });
  },

  async getDocument(userId: string, id: string) {
    return prisma.document.findFirst({
      where: { id, userId },
      include: { ingestion: true },
    });
  },

  async reingest(userId: string, id: string) {
    const document = await prisma.document.findFirst({
      where: { id, userId },
    });
    if (!document) {
      return null;
    }
    void ingestDocument(document.id);
    return document;
  },
};
