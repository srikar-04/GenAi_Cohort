import { z } from 'zod';

export const DocumentTypeSchema = z.enum(['GENERAL', 'RESEARCH']);
export type DocumentType = z.infer<typeof DocumentTypeSchema>;

export const CreateDocumentSchema = z.object({
  fileName: z.string().min(1).max(255),
  fileSize: z.number().int().positive().max(50 * 1024 * 1024),
  mimeType: z.literal('application/pdf'),
  documentType: DocumentTypeSchema.default('GENERAL'),
});

export const PresignRequestSchema = z.object({
  fileName: z.string().min(1).max(255),
  fileSize: z.number().int().positive().max(50 * 1024 * 1024),
  mimeType: z.literal('application/pdf'),
  documentType: DocumentTypeSchema.default('GENERAL'),
});

export const RegisterUploadSchema = z.object({
  storageKey: z.string().min(1),
  storageUrl: z.string().url(),
  fileName: z.string().min(1).max(255),
  fileSize: z.number().int().positive().max(50 * 1024 * 1024),
  mimeType: z.literal('application/pdf'),
  checksum: z.string().min(16).max(128),
  documentType: DocumentTypeSchema.default('GENERAL'),
});

export const RagQuerySchema = z.object({
  documentId: z.string().uuid().optional(),
  question: z.string().min(4).max(2000),
  topK: z.number().int().positive().max(20).default(6),
  minScore: z.number().min(0).max(1).default(0.15),
});

export type RagQuery = z.infer<typeof RagQuerySchema>;
