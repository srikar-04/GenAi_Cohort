import 'dotenv/config';
import { createApp } from './app';
import { env } from './config/env';
import { logger } from './config/logger';
import { prisma } from './db/prisma';

const start = async () => {
  try {
    await prisma.$connect();
    const app = createApp();
    app.listen(env.PORT, () => {
      logger.info({ port: env.PORT }, 'Server running');
    });
  } catch (error) {
    logger.error(error, 'Failed to start server');
    process.exit(1);
  }
};

start();

process.on('SIGINT', async () => {
  await prisma.$disconnect();
  process.exit(0);
});
