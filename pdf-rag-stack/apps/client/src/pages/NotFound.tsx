import { Link } from 'react-router-dom';

export const NotFound = () => {
  return (
    <div className="notfound">
      <h2>Page not found</h2>
      <p>The page you are looking for does not exist.</p>
      <Link className="btn primary" to="/">
        Return home
      </Link>
    </div>
  );
};
