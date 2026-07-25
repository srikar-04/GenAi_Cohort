import { v2 as cloudinary } from 'cloudinary';
import https from 'https';
import { env } from '../../config/env';

cloudinary.config({
  cloud_name: env.CLOUDINARY_CLOUD_NAME,
  api_key: env.CLOUDINARY_API_KEY,
  api_secret: env.CLOUDINARY_API_SECRET,
  secure: true,
});

const sanitizeFileName = (fileName: string) =>
  fileName.replace(/[^a-zA-Z0-9._-]/g, '_').slice(0, 120);

const downloadBuffer = (url: string) =>
  new Promise<Buffer>((resolve, reject) => {
    https
      .get(url, (res) => {
        if (!res.statusCode || res.statusCode >= 400) {
          return reject(new Error(`Failed to download file: ${res.statusCode}`));
        }
        const chunks: Buffer[] = [];
        res.on('data', (chunk) => chunks.push(Buffer.from(chunk)));
        res.on('end', () => resolve(Buffer.concat(chunks)));
      })
      .on('error', reject);
  });

export const cloudinaryStorage = {
  async uploadBuffer(params: { buffer: Buffer; fileName: string; folder: string }) {
    const safeName = sanitizeFileName(params.fileName);
    const publicId = `${params.folder}/${Date.now()}-${safeName}`;

    const result = await new Promise<cloudinary.UploadApiResponse>((resolve, reject) => {
      const stream = cloudinary.uploader.upload_stream(
        {
          resource_type: 'raw',
          type: 'upload',
          access_mode: 'public',
          public_id: publicId,
          filename_override: safeName,
          use_filename: false,
          unique_filename: false,
        },
        (error, res) => {
          if (error || !res) {
            return reject(error ?? new Error('Cloudinary upload failed'));
          }
          resolve(res);
        },
      );
      stream.end(params.buffer);
    });

    return {
      key: result.public_id,
      url: result.secure_url,
    };
  },

  getSignedUploadPayload(params: { folder: string; fileName: string }) {
    const timestamp = Math.floor(Date.now() / 1000);
    const safeName = sanitizeFileName(params.fileName);
    const publicId = `${params.folder}/${Date.now()}-${safeName}`;
    const signature = cloudinary.utils.api_sign_request(
      {
        public_id: publicId,
        timestamp,
        resource_type: 'raw',
      },
      env.CLOUDINARY_API_SECRET,
    );

    return {
      uploadUrl: `https://api.cloudinary.com/v1_1/${env.CLOUDINARY_CLOUD_NAME}/raw/upload`,
      apiKey: env.CLOUDINARY_API_KEY,
      timestamp,
      signature,
      publicId,
      resourceType: 'raw',
    };
  },

  async getObjectBuffer(params: { url?: string; publicId?: string }) {
    if (params.url) {
      try {
        return await downloadBuffer(params.url);
      } catch (error) {
        const message = (error as Error).message ?? '';
        const isAuthError = message.includes('401') || message.includes('403');
        if (!isAuthError || !params.publicId) {
          throw error;
        }
      }
    }
    if (!params.publicId) {
      throw new Error('Missing Cloudinary public ID for download');
    }
    const signedUrl = cloudinary.url(params.publicId, {
      resource_type: 'raw',
      type: 'upload',
      sign_url: true,
      secure: true,
      expires_at: Math.floor(Date.now() / 1000) + 300,
    });
    return downloadBuffer(signedUrl);
  },
};
