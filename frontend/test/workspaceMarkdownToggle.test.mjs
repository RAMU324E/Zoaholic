import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

// 修改原因：工作区的 Markdown 文件需要在渲染预览和源码查看之间切换，不能只显示纯文本源码。
// 修改方式：用源码静态断言锁定 Workspace.tsx 的 MarkdownRenderer 引入、Markdown 识别、preview/code 状态和两个切换按钮。
// 目的：防止后续维护把 Markdown 文件重新退回到单一源码查看状态。
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const frontendRoot = path.resolve(__dirname, '..');
const workspaceSource = readFileSync(path.resolve(frontendRoot, 'src/pages/Workspace.tsx'), 'utf8');

function sliceBetween(source, startMarker, endMarker, fromIndex = 0) {
  const start = source.indexOf(startMarker, fromIndex);
  assert.notEqual(start, -1, `找不到起始片段：${startMarker}`);
  const end = source.indexOf(endMarker, start + startMarker.length);
  assert.notEqual(end, -1, `找不到结束片段：${endMarker}`);
  return source.slice(start, end);
}

assert.match(workspaceSource, /import \{ MarkdownRenderer \} from '\.\.\/components\/MarkdownRenderer';/, '工作区应复用统一 MarkdownRenderer 渲染 Markdown 文件');
assert.match(workspaceSource, /type MarkdownViewMode = 'preview' \| 'code';/, '工作区应有明确的 Markdown 查看模式类型');
assert.match(workspaceSource, /function isMarkdownFile\(file: Pick<FileContent, 'path' \| 'language'>\): boolean/, '工作区应集中判断 Markdown 文件，兼容 language 和扩展名');
assert.match(workspaceSource, /const \[markdownViewMode, setMarkdownViewMode\] = useState<MarkdownViewMode>\('preview'\);/, '工作区应保存 preview/code 切换状态，默认使用预览');
assert.match(workspaceSource, /setMarkdownViewMode\(isMarkdownFile\(data\) \? 'preview' : 'code'\);/, '切换文件时 Markdown 应默认打开渲染预览，非 Markdown 应回到源码查看');

const fileHeader = sliceBetween(workspaceSource, '{/* File header */}', '{saveMessage && (');
assert.match(fileHeader, /selectedFile && isMarkdownFile\(selectedFile\) && !editMode/, '只有非编辑状态的 Markdown 文件才应显示 preview/code 切换按钮');
assert.match(fileHeader, /onClick=\{\(\) => setMarkdownViewMode\('preview'\)\}/, '切换按钮应能进入渲染预览');
assert.match(fileHeader, /onClick=\{\(\) => setMarkdownViewMode\('code'\)\}/, '切换按钮应能进入源码查看');
// 修改原因：JSX 中按钮文案可能换行缩进，直接匹配 >Preview< 会误判已经存在的按钮。
// 修改方式：允许标签与文案之间出现空白字符。
// 目的：让测试关注按钮文案是否存在，而不是源码排版。
assert.match(fileHeader, />\s*Preview\s*</, '切换按钮应包含 Preview 标签');
assert.match(fileHeader, />\s*Code\s*</, '切换按钮应包含 Code 标签');

const fileContent = sliceBetween(workspaceSource.replace(/\r\n/g, '\n'), '{/* File content */}', '</div>\n            </>');
assert.match(fileContent, /isMarkdownFile\(selectedFile\) && markdownViewMode === 'preview'/, 'Markdown 文件在 preview 模式应走渲染分支');
assert.match(fileContent, /<MarkdownRenderer[\s\S]*content=\{selectedFile\.content\}/, '渲染分支应传入当前文件内容');
assert.match(fileContent, /<pre[\s\S]*\{selectedFile\.content\}/, '源码查看分支应保留原始内容显示');

console.log('workspace markdown toggle regression passed');
// 修改原因：当前部署环境的 Node 18 在部分 ESM 脚本自然结束后可能触发 Aborted。
// 修改方式：断言全部通过后显式以 0 退出，断言失败时仍会在这里之前抛出错误。
// 目的：让测试退出码只反映本文件断言是否通过。
process.exit(0);
