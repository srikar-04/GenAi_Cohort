import { gemini } from '../../integrations/gemini';
import { embedQuery } from '../../integrations/embeddings/gemini-embeddings';
import { similaritySearch } from '../../integrations/vector/pgvector';
import { env } from '../../config/env';

const SYSTEM_PROMPT = `You are a precise research assistant. Use only the provided context.
If the answer is not in the context, say "I don't know based on the provided document."`;

export const ragService = {
  async answerQuestion(params: {
    userId: string;
    documentId?: string;
    question: string;
    topK: number;
    minScore: number;
  }) {
    const queryEmbedding = await embedQuery(params.question);
    const chunks = await similaritySearch({
      userId: params.userId,
      documentId: params.documentId,
      embedding: queryEmbedding,
      topK: params.topK,
      minScore: params.minScore,
    });

    if (chunks.length === 0) {
      return {
        answer: "I don't know based on the provided document.",
        sources: [],
      };
    }

    let context = chunks
      .map((chunk, index) => {
        const section = (chunk.metadata as any)?.section ?? 'Section';
        return `Source ${index + 1} (${section}):\n${chunk.content}`;
      })
      .join('\n\n');

    if (context.length > 12000) {
      context = `${context.slice(0, 12000)}\n\n[Context truncated for length]`;
    }

    const model = gemini.getGenerativeModel({
      model: env.GEMINI_CHAT_MODEL,
      systemInstruction: SYSTEM_PROMPT,
    });
    const response = await model.generateContent(
      `Question: ${params.question}\n\nContext:\n${context}`,
    );

    return {
      answer: response.response.text(),
      sources: chunks.map((chunk) => ({
        id: chunk.id,
        documentId: chunk.documentId,
        score: chunk.score,
        metadata: chunk.metadata,
      })),
    };
  },
};
