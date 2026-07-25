import { NextFunction, Request, Response } from 'express';
import { ZodError } from 'zod';
import { AppError } from '../errors';
import { logger } from '../../config/logger';

export const errorHandler = (err: Error, req: Request, res: Response, _next: NextFunction) => {
  if (err instanceof ZodError) {
    return res.status(400).json({
      error: 'ValidationError',
      details: err.flatten(),
      requestId: req.requestId,
    });
  }

  if (err instanceof AppError) {
    return res.status(err.status).json({
      error: err.code ?? 'AppError',
      message: err.message,
      requestId: req.requestId,
    });
  }

  logger.error({ err, requestId: req.requestId }, 'Unhandled error');
  return res.status(500).json({
    error: 'InternalServerError',
    message: 'Something went wrong.',
    requestId: req.requestId,
  });
};
