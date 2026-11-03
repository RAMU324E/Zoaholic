/**
 * Split pasted text into API keys.
 *
 * Why: the key input splits pastes by line, so a multi-line service account
 * JSON (Vertex) became one broken key per line. How: if the paste parses as a
 * JSON object (or an array of objects), each object becomes one compact
 * single-line key; anything else keeps the one-key-per-line behaviour.
 */
export function splitPastedKeys(text: string): string[] {
  // Tolerate a trailing comma left over from copying one element out of an array.
  const trimmed = text.trim().replace(/,\s*$/, '');
  if (trimmed.startsWith('{') || trimmed.startsWith('[')) {
    try {
      const parsed: unknown = JSON.parse(trimmed);
      const items = Array.isArray(parsed) ? parsed : [parsed];
      if (items.length > 0 && items.every(item => item !== null && typeof item === 'object' && !Array.isArray(item))) {
        return items.map(item => JSON.stringify(item));
      }
    } catch {
      // Not JSON: fall back to line splitting.
    }
  }
  return text.split(/\r?\n|\r/).map(s => s.trim()).filter(Boolean);
}
