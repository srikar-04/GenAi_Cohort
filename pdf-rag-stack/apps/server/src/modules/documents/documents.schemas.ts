import { z } from 'zod';
import { DocumentTypeSchema, PresignRequestSchema, RegisterUploadSchema } from '@pdf-rag/contracts';

export const DocumentUploadSchema = z.object({
  documentType: DocumentTypeSchema.optional(),
  title: z.string().min(1).max(200).optional(),
});

export const DocumentListQuerySchema = z.object({
  status: z.enum(['UPLOADED', 'PROCESSING', 'READY', 'FAILED']).optional(),
});

export { PresignRequestSchema, RegisterUploadSchema };
