import { NextFunction, Request, Response } from 'express';
import { AppError } from '../errors';

export const requireAuth = (req: Request, _res: Response, next: NextFunction) => {
  if (!req.isAuthenticated || !req.isAuthenticated()) {
    return next(new AppError('Authentication required', 401, 'Unauthorized'));
  }
  return next();
};
