import { AuthProvider, Prisma } from '@prisma/client';
import type { Profile as GitHubProfile } from 'passport-github2';
import type { Profile as GoogleProfile } from 'passport-google-oauth20';
import { prisma } from '../../db/prisma';

type OAuthProfile = GitHubProfile | GoogleProfile;

const getProfileEmail = (profile: OAuthProfile) => {
  return profile.emails && profile.emails.length > 0 ? profile.emails[0].value : null;
};

const getProfileImage = (profile: OAuthProfile) => {
  return profile.photos && profile.photos.length > 0 ? profile.photos[0].value : null;
};

const getProfileName = (profile: OAuthProfile) => {
  return profile.displayName || profile.username || getProfileEmail(profile) || null;
};

export const authService = {
  async upsertOAuthUser(params: {
    provider: AuthProvider;
    providerUserId: string;
    profile: OAuthProfile;
    accessToken?: string;
    refreshToken?: string;
  }) {
    const { provider, providerUserId, profile, accessToken, refreshToken } = params;
    const email = getProfileEmail(profile);
    const imageUrl = getProfileImage(profile);
    const name = getProfileName(profile);

    const existingAccount = await prisma.oAuthAccount.findUnique({
      where: {
        provider_providerUserId: {
          provider,
          providerUserId,
        },
      },
      include: { user: true },
    });

    if (existingAccount) {
      await prisma.oAuthAccount.update({
        where: { id: existingAccount.id },
        data: {
          accessToken,
          refreshToken,
        },
      });

      const user = await prisma.user.update({
        where: { id: existingAccount.userId },
        data: {
          email: email ?? existingAccount.user.email,
          name: name ?? existingAccount.user.name,
          imageUrl: imageUrl ?? existingAccount.user.imageUrl,
        },
      });

      return user;
    }

    const connectOrCreateUser: Prisma.UserCreateNestedOneWithoutAccountsInput = email
      ? {
          connectOrCreate: {
            where: { email },
            create: {
              email,
              name,
              imageUrl,
            },
          },
        }
      : {
          create: {
            name,
            imageUrl,
          },
        };

    const account = await prisma.oAuthAccount.create({
      data: {
        provider,
        providerUserId,
        accessToken,
        refreshToken,
        user: connectOrCreateUser,
      },
      include: { user: true },
    });

    return account.user;
  },
};
