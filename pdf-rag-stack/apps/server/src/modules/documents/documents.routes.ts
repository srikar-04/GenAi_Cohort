import { Router } from 'express';
import multer from 'multer';
import { requireAuth } from '../../common/middleware/requireAuth';
import { AppError } from '../../common/errors';
import { env } from '../../config/env';
import { validate } from '../../common/validation';
import {
  DocumentListQuerySchema,
  DocumentUploadSchema,
  PresignRequestSchema,
  RegisterUploadSchema,
} from './documents.schemas';
import { documentsService } from './documents.service';
import { cloudinaryStorage } from '../../integrations/storage/cloudinary';

export const documentRoutes = Router();

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: env.MAX_UPLOAD_MB * 1024 * 1024 },
  fileFilter: (_req, file, cb) => {
    if (file.mimetype !== 'application/pdf') {
      return cb(new AppError('Only PDF files are allowed', 400));
    }
    cb(null, true);
  },
});

documentRoutes.get('/', requireAuth, validate(DocumentListQuerySchema, 'query'), async (req, res) => {
  const docs = await documentsService.listDocuments(
    req.user!.id,
    req.query.status as any,
  );
  res.json({ documents: docs });
});

documentRoutes.get('/:id', requireAuth, async (req, res, next) => {
  const doc = await documentsService.getDocument(req.user!.id, req.params.id);
  if (!doc) {
    return next(new AppError('Document not found', 404));
  }
  res.json({ document: doc });
});

documentRoutes.post(
  '/upload',
  requireAuth,
  upload.single('file'),
  async (req, res, next) => {
    if (!req.file) {
      return next(new AppError('Missing file', 400));
    }
    const parsed = DocumentUploadSchema.safeParse(req.body);
    if (!parsed.success) {
      return next(parsed.error);
    }
    const document = await documentsService.createFromUpload({
      userId: req.user!.id,
      fileName: req.file.originalname,
      mimeType: req.file.mimetype,
      fileSize: req.file.size,
      buffer: req.file.buffer,
      documentType: parsed.data.documentType,
      title: parsed.data.title,
    });
    res.status(201).json({ document });
  },
);

documentRoutes.post('/presign', requireAuth, validate(PresignRequestSchema), async (req, res) => {
  const payload = cloudinaryStorage.getSignedUploadPayload({
    folder: req.user!.id,
    fileName: req.body.fileName,
  });
  res.json(payload);
});

documentRoutes.post('/register', requireAuth, validate(RegisterUploadSchema), async (req, res) => {
  const document = await documentsService.registerPresignedUpload({
    userId: req.user!.id,
    fileName: req.body.fileName,
    mimeType: req.body.mimeType,
    fileSize: req.body.fileSize,
    storageKey: req.body.storageKey,
    storageUrl: req.body.storageUrl,
    checksum: req.body.checksum,
    documentType: req.body.documentType,
  });
  res.status(201).json({ document });
});

documentRoutes.post('/:id/ingest', requireAuth, async (req, res, next) => {
  const document = await documentsService.reingest(req.user!.id, req.params.id);
  if (!document) {
    return next(new AppError('Document not found', 404));
  }
  res.json({ ok: true });
});
