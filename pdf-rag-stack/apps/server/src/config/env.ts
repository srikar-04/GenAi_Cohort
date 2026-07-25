import { z } from 'zod';

const envSchema = z.object({
  NODE_ENV: z.enum(['development', 'test', 'production']).default('development'),
  PORT: z.coerce.number().default(4000),
  APP_URL: z.string().url(),
  API_URL: z.string().url(),
  SESSION_SECRET: z.string().min(16),
  DATABASE_URL: z.string().url(),
  OAUTH_GITHUB_CLIENT_ID: z.string().optional(),
  OAUTH_GITHUB_CLIENT_SECRET: z.string().optional(),
  OAUTH_GOOGLE_CLIENT_ID: z.string().optional(),
  OAUTH_GOOGLE_CLIENT_SECRET: z.string().optional(),
  OAUTH_CALLBACK_BASE: z.string().url(),
  CLOUDINARY_CLOUD_NAME: z.string(),
  CLOUDINARY_API_KEY: z.string(),
  CLOUDINARY_API_SECRET: z.string(),
  GEMINI_API_KEY: z.string(),
  GEMINI_EMBEDDING_MODEL: z.string().default('text-embedding-004'),
  GEMINI_CHAT_MODEL: z.string().default('gemini-1.5-flash'),
  MAX_UPLOAD_MB: z.coerce.number().default(50),
  RAG_TOP_K: z.coerce.number().default(6),
  RAG_MIN_SCORE: z.coerce.number().default(0.15),
});

export const env = envSchema.parse(process.env);
