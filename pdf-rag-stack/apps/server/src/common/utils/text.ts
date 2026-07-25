export const normalizeWhitespace = (text: string) =>
  text
    .replace(/\r\n/g, '\n')
    .replace(/[ \t]+/g, ' ')
    .replace(/\n{3,}/g, '\n\n')
    .trim();

export const stripHyphenation = (text: string) => text.replace(/-\n([a-z])/gi, '$1');

export const estimateTokens = (text: string) =>
  Math.max(1, Math.ceil(text.split(/\s+/).length * 1.3));

export const splitByHeading = (text: string, headings: string[]) => {
  const pattern = new RegExp(`^(${headings.join('|')})\\s*$`, 'im');
  const parts: { heading: string; content: string }[] = [];
  let current = { heading: 'UNKNOWN', content: '' };

  for (const line of text.split('\n')) {
    if (pattern.test(line.trim())) {
      if (current.content.trim()) {
        parts.push(current);
      }
      current = { heading: line.trim(), content: '' };
    } else {
      current.content += `${line}\n`;
    }
  }
  if (current.content.trim()) {
    parts.push(current);
  }

  return parts;
};
