import { normalizeWhitespace, splitByHeading } from '../../common/utils/text';
import { chunkSections, Chunk } from './chunker';

const RESEARCH_HEADINGS = [
  'Abstract',
  'Introduction',
  'Related Work',
  'Background',
  'Methods',
  'Methodology',
  'Materials and Methods',
  'Experiments',
  'Results',
  'Discussion',
  'Conclusion',
  'References',
  'Acknowledgements',
  'Appendix',
];

const stripReferences = (text: string) => {
  const lower = text.toLowerCase();
  const idx = lower.lastIndexOf('\nreferences');
  if (idx > 0) {
    return text.slice(0, idx);
  }
  return text;
};

export const chunkResearchPaper = (pages: string[], options: { chunkSize: number; overlap: number }) => {
  const raw = normalizeWhitespace(pages.join('\n\n'));
  const cleaned = stripReferences(raw);
  const sections = splitByHeading(cleaned, RESEARCH_HEADINGS);
  const chunks = chunkSections(sections, options);

  return chunks.map((chunk) => ({
    ...chunk,
    metadata: {
      ...(chunk.metadata ?? {}),
      documentType: 'RESEARCH',
    },
  })) satisfies Chunk[];
};
