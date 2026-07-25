import { NextFunction, Request, Response } from 'express';
import { randomUUID } from 'crypto';

export const attachRequestId = () => (req: Request, res: Response, next: NextFunction) => {
  const headerId = req.header('x-request-id');
  req.requestId = headerId ?? randomUUID();
  res.setHeader('x-request-id', req.requestId);
  next();
};
