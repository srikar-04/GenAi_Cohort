import { createSlice, PayloadAction } from '@reduxjs/toolkit';

type User = {
  id: string;
  email?: string | null;
  name?: string | null;
  imageUrl?: string | null;
};

type AuthState = {
  user: User | null;
  loaded: boolean;
};

const initialState: AuthState = {
  user: null,
  loaded: false,
};

const authSlice = createSlice({
  name: 'auth',
  initialState,
  reducers: {
    setUser(state, action: PayloadAction<User | null>) {
      state.user = action.payload;
      state.loaded = true;
    },
    clearUser(state) {
      state.user = null;
      state.loaded = true;
    },
  },
});

export const { setUser, clearUser } = authSlice.actions;
export default authSlice.reducer;
