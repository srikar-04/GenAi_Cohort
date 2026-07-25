import 'express-async-errors';
import express from 'express';
import helmet from 'helmet';
import cors from 'cors';
import compression from 'compression';
import cookieParser from 'cookie-parser';
import session from 'express-session';
import rateLimit from 'express-rate-limit';
import pinoHttp from 'pino-http';

import { env } from './config/env';
import { logger } from './config/logger';
import { attachRequestId } from './common/middleware/requestId';
import { errorHandler } from './common/middleware/errorHandler';
import { notFoundHandler } from './common/middleware/notFound';
import { initPassport } from './modules/auth/passport';
import { routes } from './routes';

export const createApp = () => {
  const app = express();

  app.set('trust proxy', 1);
  app.use(helmet());
  app.use(
    cors({
      origin: env.APP_URL,
      credentials: true,
    }),
  );
  app.use(compression());
  app.use(express.json({ limit: `${env.MAX_UPLOAD_MB}mb` }));
  app.use(express.urlencoded({ extended: true }));
  app.use(cookieParser());
  app.use(attachRequestId());
  app.use(
    pinoHttp({
      logger,
      customProps: (req) => ({ requestId: req.requestId }),
    }),
  );
  app.use(
    rateLimit({
      windowMs: 60 * 1000,
      max: env.NODE_ENV === 'production' ? 120 : 1000,
      standardHeaders: true,
      legacyHeaders: false,
    }),
  );

  app.use(
    session({
      name: 'pdf_rag_session',
      secret: env.SESSION_SECRET,
      resave: false,
      saveUninitialized: false,
      cookie: {
        httpOnly: true,
        sameSite: 'lax',
        secure: env.NODE_ENV === 'production',
        maxAge: 1000 * 60 * 60 * 24 * 7,
      },
    }),
  );

  initPassport(app);
  app.use('/api', routes);

  app.use(notFoundHandler);
  app.use(errorHandler);

  return app;
};
