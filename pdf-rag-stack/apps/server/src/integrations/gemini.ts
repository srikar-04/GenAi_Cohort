import { GoogleGenerativeAI } from '@google/generative-ai';
import { env } from '../config/env';

export const gemini = new GoogleGenerativeAI(env.GEMINI_API_KEY);
