# PDF RAG Stack

Full-stack PDF RAG monorepo (React + Node + TypeScript) designed for production scale.

## What's Inside
- `apps/server`: Express + Prisma + Postgres + pgvector backend (Cloudinary + Gemini)
- `apps/client`: Vite + React + Redux Toolkit frontend
- `packages/contracts`: Shared Zod schemas and types

## Quick Start
1. Install dependencies
   - `pnpm install`
2. Configure env files
   - `apps/server/.env.example` -> `apps/server/.env`
   - `apps/client/.env.example` -> `apps/client/.env`
3. Start Postgres and enable pgvector
   - Apply `apps/server/prisma/pgvector.sql` after running Prisma migrations.
4. Run dev servers
   - `pnpm dev` (builds shared contracts first)

## RAG Flow (Server)
1. Validate upload metadata and auth session (Zod + OAuth)
2. Upload PDF to Cloudinary
3. Extract text per page (PDF.js)
4. Normalize and chunk (section-aware for research papers)
5. Embed chunks (Gemini embeddings)
6. Upsert into Postgres + pgvector
7. Retrieve top-k chunks and answer with citations

## Production Notes
- Switch session storage to Redis in `apps/server/src/app.ts`
- Use a job queue (BullMQ) to process ingestion asynchronously
- Add content moderation and cost-based throttling
- Separate vector storage if chunk volumes grow beyond Postgres
