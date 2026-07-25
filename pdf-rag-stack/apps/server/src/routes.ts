import { Router } from 'express';
import { authRoutes } from './modules/auth/auth.routes';
import { documentRoutes } from './modules/documents/documents.routes';
import { ragRoutes } from './modules/rag/rag.routes';
import { healthRoutes } from './modules/health/health.routes';

export const routes = Router();

routes.use('/health', healthRoutes);
routes.use('/auth', authRoutes);
routes.use('/documents', documentRoutes);
routes.use('/rag', ragRoutes);
