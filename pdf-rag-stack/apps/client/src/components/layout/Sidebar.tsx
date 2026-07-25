import { NavLink } from 'react-router-dom';
import { useAppSelector } from '../../store/hooks';

export const Sidebar = () => {
  const user = useAppSelector((state) => state.auth.user);
  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <div className="brand-mark">R</div>
        <div>
          <div className="brand-title">PDF RAG</div>
          <div className="brand-subtitle">Research workspace</div>
        </div>
      </div>
      <nav className="sidebar-nav">
        <NavLink to="/app/dashboard">Dashboard</NavLink>
        <NavLink to="/app/documents">Documents</NavLink>
        <NavLink to="/app/chat">Chat</NavLink>
      </nav>
      <div className="sidebar-footer">
        <div className="user-card">
          <div className="user-avatar">{user?.name?.slice(0, 1) ?? 'U'}</div>
          <div>
            <div className="user-name">{user?.name ?? 'User'}</div>
            <div className="user-email">{user?.email ?? 'Signed in'}</div>
          </div>
        </div>
      </div>
    </aside>
  );
};
