import { Request, Response, NextFunction } from 'express';
import { z, ZodSchema } from 'zod';

export const validate =
  <T extends ZodSchema>(schema: T, source: 'body' | 'query' | 'params' = 'body') =>
  (req: Request, _res: Response, next: NextFunction) => {
    const result = schema.safeParse(req[source]);
    if (!result.success) {
      return next(result.error);
    }
    req[source] = result.data as typeof req[typeof source];
    return next();
  };

export const parseJson = <T>(schema: z.ZodType<T>, payload: unknown) => schema.parse(payload);
