export {};

declare global {
  namespace Express {
    interface User {
      id: string;
      email?: string | null;
      name?: string | null;
      imageUrl?: string | null;
    }

    interface Request {
      requestId?: string;
    }
  }
}
