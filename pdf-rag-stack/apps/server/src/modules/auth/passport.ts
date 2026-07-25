import { Express } from 'express';
import passport from 'passport';
import { Strategy as GitHubStrategy } from 'passport-github2';
import { Strategy as GoogleStrategy } from 'passport-google-oauth20';
import { AuthProvider } from '@prisma/client';

import { env } from '../../config/env';
import { authService } from './auth.service';
import { prisma } from '../../db/prisma';

export const initPassport = (app: Express) => {
  passport.serializeUser((user, done) => {
    done(null, (user as Express.User).id);
  });

  passport.deserializeUser(async (id: string, done) => {
    try {
      const user = await prisma.user.findUnique({ where: { id } });
      if (!user) {
        return done(null, false);
      }
      return done(null, {
        id: user.id,
        email: user.email,
        name: user.name,
        imageUrl: user.imageUrl,
      });
    } catch (error) {
      return done(error);
    }
  });

  if (env.OAUTH_GITHUB_CLIENT_ID && env.OAUTH_GITHUB_CLIENT_SECRET) {
    passport.use(
      new GitHubStrategy(
        {
          clientID: env.OAUTH_GITHUB_CLIENT_ID,
          clientSecret: env.OAUTH_GITHUB_CLIENT_SECRET,
          callbackURL: `${env.OAUTH_CALLBACK_BASE}/api/auth/github/callback`,
          scope: ['user:email'],
        },
        async (accessToken, refreshToken, profile, done) => {
          try {
            const user = await authService.upsertOAuthUser({
              provider: AuthProvider.GITHUB,
              providerUserId: profile.id,
              profile,
              accessToken,
              refreshToken,
            });
            done(null, user);
          } catch (error) {
            done(error as Error);
          }
        },
      ),
    );
  }

  if (env.OAUTH_GOOGLE_CLIENT_ID && env.OAUTH_GOOGLE_CLIENT_SECRET) {
    passport.use(
      new GoogleStrategy(
        {
          clientID: env.OAUTH_GOOGLE_CLIENT_ID,
          clientSecret: env.OAUTH_GOOGLE_CLIENT_SECRET,
          callbackURL: `${env.OAUTH_CALLBACK_BASE}/api/auth/google/callback`,
        },
        async (accessToken, refreshToken, profile, done) => {
          try {
            const user = await authService.upsertOAuthUser({
              provider: AuthProvider.GOOGLE,
              providerUserId: profile.id,
              profile,
              accessToken,
              refreshToken,
            });
            done(null, user);
          } catch (error) {
            done(error as Error);
          }
        },
      ),
    );
  }

  app.use(passport.initialize());
  app.use(passport.session());
};
