import { useLogoutMutation } from '../../features/auth/authApi';
import { clearUser } from '../../features/auth/authSlice';
import { useAppDispatch } from '../../store/hooks';

export const Topbar = () => {
  const dispatch = useAppDispatch();
  const [logout, { isLoading }] = useLogoutMutation();

  const handleLogout = async () => {
    await logout();
    dispatch(clearUser());
  };

  return (
    <header className="topbar">
      <div className="topbar-title">Your RAG Workspace</div>
      <div className="topbar-actions">
        <button className="btn ghost" onClick={handleLogout} disabled={isLoading}>
          Sign out
        </button>
      </div>
    </header>
  );
};
