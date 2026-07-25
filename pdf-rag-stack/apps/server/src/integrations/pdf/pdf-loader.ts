import * as pdfjsLib from 'pdfjs-dist/legacy/build/pdf.mjs';
import { normalizeWhitespace, stripHyphenation } from '../../common/utils/text';

pdfjsLib.GlobalWorkerOptions.workerSrc = '';

export const loadPdfText = async (buffer: Buffer) => {
  const data = new Uint8Array(buffer);
  const doc = await pdfjsLib.getDocument({ data }).promise;

  const pages: string[] = [];
  for (let i = 1; i <= doc.numPages; i += 1) {
    const page = await doc.getPage(i);
    const content = await page.getTextContent();
    const text = content.items.map((item: any) => item.str).join(' ');
    const cleaned = normalizeWhitespace(stripHyphenation(text));
    pages.push(cleaned);
  }

  return {
    pages,
    pageCount: doc.numPages,
  };
};
