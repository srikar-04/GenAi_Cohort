import { estimateTokens } from '../../common/utils/text';

export type Chunk = {
  content: string;
  tokenCount: number;
  metadata?: Record<string, unknown>;
};

export const chunkText = (
  text: string,
  options: { chunkSize: number; overlap: number; metadata?: Record<string, unknown> },
): Chunk[] => {
  const words = text.split(/\s+/).filter(Boolean);
  const chunks: Chunk[] = [];
  const { chunkSize, overlap } = options;
  let start = 0;

  while (start < words.length) {
    const end = Math.min(words.length, start + chunkSize);
    const slice = words.slice(start, end).join(' ');
    const content = slice.trim();
    if (content.length > 0) {
      chunks.push({
        content,
        tokenCount: estimateTokens(content),
        metadata: options.metadata,
      });
    }
    if (end === words.length) {
      break;
    }
    start = end - overlap;
  }

  return chunks;
};

export const chunkSections = (
  sections: { heading: string; content: string }[],
  options: { chunkSize: number; overlap: number },
): Chunk[] => {
  const chunks: Chunk[] = [];
  for (const section of sections) {
    const sectionChunks = chunkText(section.content, {
      chunkSize: options.chunkSize,
      overlap: options.overlap,
      metadata: { section: section.heading },
    });
    chunks.push(...sectionChunks);
  }
  return chunks;
};
