import { Router } from 'express';
import { validate } from '../../common/validation';
import { requireAuth } from '../../common/middleware/requireAuth';
import { RagQuerySchema } from './rag.schemas';
import { ragService } from './rag.service';

export const ragRoutes = Router();

ragRoutes.post('/query', requireAuth, validate(RagQuerySchema), async (req, res) => {
  const response = await ragService.answerQuestion({
    userId: req.user!.id,
    documentId: req.body.documentId,
    question: req.body.question,
    topK: req.body.topK,
    minScore: req.body.minScore,
  });
  res.json(response);
});
