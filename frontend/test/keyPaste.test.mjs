import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { test } from 'node:test';
import { fileURLToPath, pathToFileURL } from 'node:url';

const frontendRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const helperPath = 'src/pages/channels/lib/keyPaste.ts';
const tempDir = mkdtempSync(path.join(os.tmpdir(), 'zoaholic-key-paste-'));
let splitPastedKeys;
try {
  writeFileSync(path.join(tempDir, 'package.json'), '{"type":"module"}\n');
  execFileSync(process.execPath, [
    path.join(frontendRoot, 'node_modules/typescript/bin/tsc'),
    '--target', 'ES2020', '--module', 'ES2020', '--moduleResolution', 'Bundler',
    '--rootDir', path.join(frontendRoot, 'src'), '--outDir', tempDir,
    '--noEmit', 'false', '--skipLibCheck', 'true', '--strict', 'true',
    path.join(frontendRoot, helperPath),
  ], { cwd: frontendRoot, stdio: 'pipe' });
  ({ splitPastedKeys } = await import(pathToFileURL(path.join(tempDir, 'pages/channels/lib/keyPaste.js')).href));
} finally {
  rmSync(tempDir, { recursive: true, force: true });
}

const sa = {
  type: 'service_account',
  project_id: 'demo-project',
  private_key: '-----BEGIN PRIVATE KEY-----\nABC\n-----END PRIVATE KEY-----\n',
  client_email: 'bot@demo-project.iam.gserviceaccount.com',
};

test('multi-line service account JSON becomes one compact key', () => {
  const keys = splitPastedKeys(JSON.stringify(sa, null, 2));
  assert.equal(keys.length, 1);
  assert.equal(keys[0].includes('\n'), false);
  assert.deepEqual(JSON.parse(keys[0]), sa);
});

test('array of objects and trailing comma are accepted', () => {
  const other = { ...sa, client_email: 'two@demo-project.iam.gserviceaccount.com' };
  assert.deepEqual(splitPastedKeys(JSON.stringify([sa, other], null, 2)).map(k => JSON.parse(k)), [sa, other]);
  assert.deepEqual(splitPastedKeys(`${JSON.stringify(sa, null, 2)},\n`).map(k => JSON.parse(k)), [sa]);
});

test('plain keys keep one-key-per-line splitting', () => {
  assert.deepEqual(splitPastedKeys('sk-a\r\n  sk-b \n\nsk-c'), ['sk-a', 'sk-b', 'sk-c']);
  assert.deepEqual(splitPastedKeys('sk-single'), ['sk-single']);
  assert.deepEqual(splitPastedKeys('{not json\nsk-x'), ['{not json', 'sk-x']);
  assert.deepEqual(splitPastedKeys('[1, 2]'), ['[1, 2]']);
});
