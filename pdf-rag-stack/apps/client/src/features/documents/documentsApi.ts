import { api } from '../api/api';
import { DocumentType } from '@pdf-rag/contracts';

export type Document = {
  id: string;
  title: string;
  fileName: string;
  status: 'UPLOADED' | 'PROCESSING' | 'READY' | 'FAILED';
  documentType: DocumentType;
  createdAt: string;
};

export const documentsApi = api.injectEndpoints({
  endpoints: (builder) => ({
    listDocuments: builder.query<{ documents: Document[] }, void>({
      query: () => '/documents',
      providesTags: ['Documents'],
    }),
    uploadDocument: builder.mutation<{ document: Document }, FormData>({
      query: (body) => ({
        url: '/documents/upload',
        method: 'POST',
        body,
      }),
      invalidatesTags: ['Documents'],
    }),
    reingestDocument: builder.mutation<{ ok: boolean }, string>({
      query: (id) => ({
        url: `/documents/${id}/ingest`,
        method: 'POST',
      }),
      invalidatesTags: ['Documents'],
    }),
  }),
});

export const {
  useListDocumentsQuery,
  useUploadDocumentMutation,
  useReingestDocumentMutation,
} = documentsApi;
