import { Navigate, Outlet } from 'react-router-dom';
import { useEffect } from 'react';
import { useGetMeQuery } from '../features/auth/authApi';
import { useAppDispatch, useAppSelector } from '../store/hooks';
import { clearUser, setUser } from '../features/auth/authSlice';
import { FullPageLoader } from '../components/shared/FullPageLoader';

export const ProtectedRoute = () => {
  const dispatch = useAppDispatch();
  const { data, isLoading, isError } = useGetMeQuery();
  const user = useAppSelector((state) => state.auth.user);

  useEffect(() => {
    if (data?.user) {
      dispatch(setUser(data.user));
    } else if (isError) {
      dispatch(clearUser());
    }
  }, [data, isError, dispatch]);

  if (isLoading) {
    return <FullPageLoader />;
  }

  if (!user) {
    return <Navigate to="/login" replace />;
  }

  return <Outlet />;
};
