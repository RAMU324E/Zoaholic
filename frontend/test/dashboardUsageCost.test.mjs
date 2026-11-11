import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import ts from 'typescript';

const source = readFileSync(new URL('../src/pages/Dashboard.tsx', import.meta.url), 'utf8');
const backend = readFileSync(new URL('../../routes/stats.py', import.meta.url), 'utf8');
assert.match(backend, /return float\(rec_cost or 0\)/, '汇总费用沿用历史价格，不用当前报价回填零价用量');
const data = [
  { provider: 'a', model: 'model-a', recorded_cost_usd: 0.002, cost_usd: 0.002 },
  { provider: 'a:openai', model: 'model-b', recorded_cost_usd: 0.003, cost_usd: 0.003 },
];

// 执行实际查询函数，确认费用与用量使用同一组时间、渠道及模型筛选。
const start = source.indexOf('  const fetchDashboard = async');
const end = source.indexOf('\n  useEffect(', start);
assert.ok(start >= 0 && end > start);
const { outputText } = ts.transpileModule(source.slice(start, end), {
  compilerOptions: { target: ts.ScriptTarget.ES2022 },
});
const createFetch = new Function('env', `
  const { token, rangeKey, customStart, customEnd, selectedModels, selectedProviders, selectedKeys,
    apiFetch, setCrossBreakdown, setSummary } = env;
  const RANGE_OPTIONS = [{ key: '24h', hours: 24 }];
  const noop = () => {};
  const setLoading = noop, setTimeseries = noop, setModelTimeseries = noop, setKeyTimeseries = noop,
    setTopModels = noop, setTopKeys = noop, setAvailableModels = noop, setAvailableProviders = noop,
    setAvailableKeys = noop, setRecentRequests = noop, setRecentTotal = noop, setHeatmap = noop,
    setModelBreakdown = noop, setProviderBreakdown = noop, setKeyBreakdown = noop,
    setKeyModelBreakdown = noop, setStats = noop;
  ${outputText}
  return fetchDashboard;
`);
for (const customTime of [false, true]) {
  const updates = [], summaries = [], urls = [];
  await createFetch({
    token: 'test', rangeKey: '24h', selectedProviders: ['a'], selectedModels: ['model-a', 'model-b'],
    selectedKeys: ['fake-safe-hash'],
    customStart: customTime ? '2026-10-05T00:00:00Z' : '',
    customEnd: customTime ? '2026-10-07T00:00:00Z' : '',
    setCrossBreakdown: value => updates.push(value),
    setSummary: value => summaries.push(value),
    apiFetch: async url => {
      urls.push(new URL(url, 'http://localhost'));
      return Response.json({ summary: { recorded_cost_usd: 0.005, total_cost_usd: 0.005 }, cross_breakdown: data });
    },
  })();
  assert.deepEqual(updates, [data], '统一查询应展示同一筛选下的明细');
  assert.equal(summaries[0].recorded_cost_usd, 0.005, '费用使用统一查询返回的历史价格');
  const params = urls.find(url => url.pathname.endsWith('/dashboard_overview')).searchParams;
  assert.equal(params.get('provider'), 'a'); // backend expands root/subchannel matching
  assert.equal(params.get('model'), 'model-a,model-b');
  assert.equal(params.get('key_hash'), 'fake-safe-hash');
  assert.equal(params.get('range_key'), '24h');
  if (customTime) {
    assert.equal(params.get('start_datetime'), '2026-10-05T00:00:00.000Z');
    assert.equal(params.get('end_datetime'), '2026-10-07T00:00:00.000Z');
  }
}
assert.match(source, /summary\?\.total_cost_usd/, '应显示统一筛选后的费用合计');
assert.match(source, /formatCost\(entry\.recorded_cost_usd\)/, 'Key和模型行显示历史记录费用');
console.log('dashboard filtered usage cost checks passed');
