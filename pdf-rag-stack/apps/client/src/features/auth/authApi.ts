import { api } from '../api/api';

export const authApi = api.injectEndpoints({
  endpoints: (builder) => ({
    getMe: builder.query<{ user: any }, void>({
      query: () => '/auth/me',
      providesTags: ['Auth'],
    }),
    logout: builder.mutation<{ ok: boolean }, void>({
      query: () => ({
        url: '/auth/logout',
        method: 'POST',
      }),
      invalidatesTags: ['Auth'],
    }),
  }),
});

export const { useGetMeQuery, useLogoutMutation } = authApi;
