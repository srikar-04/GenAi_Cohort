const apiUrl = import.meta.env.VITE_API_URL ?? 'http://localhost:4000';

export const Login = () => {
  return (
    <div className="auth">
      <div className="auth-card">
        <h2>Welcome back</h2>
        <p>Sign in to keep your PDFs, chats, and embeddings synced.</p>
        <div className="auth-buttons">
          <a className="btn primary" href={`${apiUrl}/api/auth/google`}>
            Continue with Google
          </a>
          <a className="btn ghost" href={`${apiUrl}/api/auth/github`}>
            Continue with GitHub
          </a>
        </div>
        <div className="auth-footnote">
          We only access your basic profile and email for account linking.
        </div>
      </div>
    </div>
  );
};
