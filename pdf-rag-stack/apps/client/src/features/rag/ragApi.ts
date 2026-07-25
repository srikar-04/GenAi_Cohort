import { api } from '../api/api';

export type RagResponse = {
  answer: string;
  sources: { id: string; documentId: string; score: number; metadata: any }[];
};

export const ragApi = api.injectEndpoints({
  endpoints: (builder) => ({
    queryRag: builder.mutation<
      RagResponse,
      { documentId?: string; question: string; topK?: number; minScore?: number }
    >({
      query: (body) => ({
        url: '/rag/query',
        method: 'POST',
        body,
      }),
    }),
  }),
});

export const { useQueryRagMutation } = ragApi;
