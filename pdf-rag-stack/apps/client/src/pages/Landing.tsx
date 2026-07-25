import { Link } from 'react-router-dom';

export const Landing = () => {
  return (
    <div className="landing">
      <div className="landing-hero">
        <h1>Turn PDFs into answers you can trust.</h1>
        <p>
          Upload papers, index them in seconds, and ask precise questions with transparent
          citations.
        </p>
        <div className="landing-actions">
          <Link className="btn primary" to="/login">
            Sign in to start
          </Link>
          <a className="btn ghost" href="https://github.com">
            View architecture
          </a>
        </div>
      </div>
      <div className="landing-panel">
        <div className="panel-card">
          <h3>Research-grade parsing</h3>
          <p>Section-aware chunking, reference trimming, and metadata-rich embeddings.</p>
        </div>
        <div className="panel-card">
          <h3>Secure by default</h3>
          <p>OAuth login, strict validation, rate limiting, and least-privilege storage.</p>
        </div>
        <div className="panel-card">
          <h3>Scale-ready</h3>
          <p>Postgres + pgvector, Prisma, and a modular backend for large teams.</p>
        </div>
      </div>
    </div>
  );
};
