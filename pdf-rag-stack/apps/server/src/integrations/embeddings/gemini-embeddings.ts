import { env } from '../../config/env';
import { gemini } from '../gemini';

export const embedTexts = async (texts: string[]) => {
  if (texts.length === 0) return [];
  const model = gemini.getGenerativeModel({ model: env.GEMINI_EMBEDDING_MODEL });
  const embeddings: number[][] = [];
  for (const text of texts) {
    const result = await model.embedContent(text);
    embeddings.push(result.embedding.values);
  }
  return embeddings;
};

export const embedQuery = async (text: string) => {
  const model = gemini.getGenerativeModel({ model: env.GEMINI_EMBEDDING_MODEL });
  const result = await model.embedContent(text);
  return result.embedding.values;
};
