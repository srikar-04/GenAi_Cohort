import { Router } from 'express';
import passport from 'passport';
import { env } from '../../config/env';
import { requireAuth } from '../../common/middleware/requireAuth';

export const authRoutes = Router();

const githubEnabled = Boolean(env.OAUTH_GITHUB_CLIENT_ID && env.OAUTH_GITHUB_CLIENT_SECRET);
const googleEnabled = Boolean(env.OAUTH_GOOGLE_CLIENT_ID && env.OAUTH_GOOGLE_CLIENT_SECRET);

if (githubEnabled) {
  authRoutes.get('/github', passport.authenticate('github'));
  authRoutes.get(
    '/github/callback',
    passport.authenticate('github', {
      failureRedirect: `${env.APP_URL}/login?error=oauth`,
    }),
    (_req, res) => {
      res.redirect(`${env.APP_URL}/app`);
    },
  );
} else {
  authRoutes.get('/github', (_req, res) => res.status(404).json({ error: 'GitHub OAuth disabled' }));
  authRoutes.get('/github/callback', (_req, res) =>
    res.status(404).json({ error: 'GitHub OAuth disabled' }),
  );
}

if (googleEnabled) {
  authRoutes.get(
    '/google',
    passport.authenticate('google', { scope: ['profile', 'email'] }),
  );
  authRoutes.get(
    '/google/callback',
    passport.authenticate('google', {
      failureRedirect: `${env.APP_URL}/login?error=oauth`,
    }),
    (_req, res) => {
      res.redirect(`${env.APP_URL}/app`);
    },
  );
} else {
  authRoutes.get('/google', (_req, res) => res.status(404).json({ error: 'Google OAuth disabled' }));
  authRoutes.get('/google/callback', (_req, res) =>
    res.status(404).json({ error: 'Google OAuth disabled' }),
  );
}

authRoutes.get('/me', requireAuth, (req, res) => {
  res.json({ user: req.user });
});

authRoutes.post('/logout', requireAuth, (req, res, next) => {
  req.logout((err) => {
    if (err) {
      return next(err);
    }
    req.session?.destroy(() => {
      res.clearCookie('pdf_rag_session');
      res.json({ ok: true });
    });
  });
});
