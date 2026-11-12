/*
 * Refactoring UI — Workflow B Constraint Brief
 * 1. Content Ranking:
 *    - Primary: Sticky Command Bar (Global Model/Provider filters + Time Range + Section ScrollSpy) & 6 Hero KPI cards (Token total, Request count, Speed tok/s, Model-aware Cache Hit %, API cost CNY/USD, Success rate ring gauge) + Unified Multi-Mode Timeseries Chart.
 *    - Secondary: Recent Requests live feed (with click-to-filter), 3 Distribution Charts (Channel Success BarChart, Model Share Donut, Endpoint Share Donut), Channel Health Table, 365-day Activity Heatmap, and Unified Usage Breakdown Table (Provider×Model / Model / Provider).
 *    - Tertiary: Axis ticks, timestamps, legend captions, helper badges.
 *    - Primary Action: Active time-range & model filter pills in the sticky header.
 * 2. Tokens & System Rules:
 *    - Theme variables: --background, --card, --foreground, --muted, --muted-foreground, --border, --primary.
 *    - Typography: tabular-nums on all numeric metrics, timestamps, tables, and chart tooltips.
 *    - Group spacing: outer gap between sections (space-y-5) > inner card padding (p-5) > intra-metric gap (gap-1.5).
 */

import { useEffect, useState, useMemo } from 'react';
import {
  Activity, Cpu, BarChart3, AlertCircle, CheckCircle2,
  RefreshCw, Server, Search, X, ArrowUp, Filter, Calendar, Layers, KeyRound
} from 'lucide-react';
import { useAuthStore } from '../store/authStore';
import { apiFetch } from '../lib/api';
import { KeyAnalyticsSheet } from '../components/KeyAnalyticsSheet';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
  PieChart, Pie, Legend, LineChart, Line, CartesianGrid, ComposedChart
} from 'recharts';

interface StatData {
  time_range: string;
  channel_success_rates: { provider: string; success_rate: number; total_requests: number }[];
  model_request_counts: { model: string; count: number }[];
  endpoint_request_counts: { endpoint: string; count: number }[];
  total_cost?: number;
}

interface OverviewSummary {
  range_key: string;
  granularity: 'hour' | 'day';
  start_datetime: string;
  end_datetime: string;
  total_tokens: number;
  prompt_tokens: number;
  completion_tokens: number;
  cached_tokens: number;
  cache_creation_tokens: number;
  uncached_tokens: number;
  cache_hit_rate: number;
  total_requests: number;
  success_requests: number;
  failed_requests: number;
  success_rate: number;
  avg_speed_tok_s: number;
  peak_speed_tok_s: number;
  avg_process_time: number;
  avg_first_response_time: number;
  p90_first_response_time: number;
  total_cost_usd: number;
  total_cost_cny: number;
  recorded_cost_usd: number;
  active_keys?: number;
}

interface TimeseriesPoint {
  bucket: string;
  label: string;
  requests: number;
  success_requests: number;
  failed_requests: number;
  success_rate: number;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  cached_tokens: number;
  uncached_tokens: number;
  cache_hit_rate: number;
  speed_tok_s: number;
  avg_process_time: number;
  avg_ttft: number;
  cost_usd: number;
  cost_cny: number;
}

interface AvailableModelItem {
  model: string;
  requests: number;
  total_tokens: number;
  cache_hit_rate: number;
  speed_tok_s: number;
}

interface AvailableProviderItem {
  provider: string;
  requests: number;
  total_tokens: number;
  success_rate: number;
  cache_hit_rate: number;
}

interface AvailableKeyItem {
  key_hash: string;
  key_label: string;
  api_key_name?: string | null;
  api_key_prefix: string;
  api_key_group?: string | null;
  parent_name?: string | null;
  can_drilldown: boolean;
  requests: number;
  total_tokens: number;
  success_rate: number;
  cache_hit_rate: number;
  cost_usd: number;
  cost_cny: number;
}

interface TopKeyItem {
  key_hash: string;
  key_label: string;
}

interface RecentRequestItem {
  id: number;
  timestamp: string;
  model: string;
  provider: string;
  key_hash?: string;
  key_label: string;
  api_key_prefix?: string;
  api_key_group?: string | null;
  can_drilldown?: boolean;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  cached_tokens: number;
  process_time: number;
  first_response_time: number;
  speed_tok_s: number;
  success: boolean;
  status_code: number;
}

interface HeatmapDay {
  date: string;
  requests: number;
  total_tokens: number;
}

interface ActivityHeatmapData {
  days: HeatmapDay[];
  active_days: number;
  total_requests: number;
  total_tokens: number;
  peak_day_requests: number;
  peak_day_date: string;
}

interface BreakdownRow {
  model?: string;
  provider?: string;
  requests: number;
  success_requests: number;
  failed_requests: number;
  success_rate: number;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  cached_tokens: number;
  cache_hit_rate: number;
  speed_tok_s: number;
  avg_process_time: number;
  avg_ttft: number;
  cost_usd: number;
  cost_cny: number;
  token_share: number;
}

interface CrossBreakdownRow {
  provider: string;
  model: string;
  requests: number;
  success_rate: number;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  cached_tokens: number;
  cache_hit_rate: number;
  speed_tok_s: number;
  recorded_cost_usd: number;
  cost_usd: number;
  cost_cny: number;
}

interface KeyBreakdownRow {
  key_hash: string;
  key_label: string;
  api_key_name?: string | null;
  api_key_prefix: string;
  api_key_group?: string | null;
  parent_name?: string | null;
  can_drilldown: boolean;
  requests: number;
  success_requests: number;
  failed_requests: number;
  success_rate: number;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  cached_tokens: number;
  uncached_tokens: number;
  cache_hit_rate: number;
  speed_tok_s: number;
  avg_process_time: number;
  avg_ttft: number;
  recorded_cost_usd: number;
  cost_usd: number;
  cost_cny: number;
  token_share: number;
  cost_share: number;
  unique_models: number;
  top_model: string;
  last_used?: string | null;
}

interface KeyModelBreakdownRow {
  key_hash: string;
  key_label: string;
  api_key_name?: string | null;
  api_key_prefix: string;
  api_key_group?: string | null;
  parent_name?: string | null;
  can_drilldown: boolean;
  model: string;
  requests: number;
  success_rate: number;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  cached_tokens: number;
  cache_hit_rate: number;
  speed_tok_s: number;
  recorded_cost_usd: number;
  cost_usd: number;
  cost_cny: number;
}

const RANGE_OPTIONS = [
  { key: '1h', label: '1 小时', hours: 1 },
  { key: '6h', label: '6 小时', hours: 6 },
  { key: '24h', label: '24 小时', hours: 24 },
  { key: 'today', label: '今日', hours: 24 },
  { key: '3d', label: '近 3 日', hours: 72 },
  { key: 'week', label: '7 天', hours: 168 },
  { key: '30d', label: '30 天', hours: 720 },
  { key: 'all', label: '全部', hours: 720 },
] as const;

type RangeKey = typeof RANGE_OPTIONS[number]['key'];

const NAV_SECTIONS = [
  { id: 'section-overview', label: '总览与时序' },
  { id: 'section-key-usage', label: 'Key 用量排行' },
  { id: 'section-distributions', label: '渠道与模型分布' },
  { id: 'section-channel-health', label: '渠道健康表' },
  { id: 'section-heatmap', label: '工作回溯' },
  { id: 'section-breakdown', label: '用量明细' },
] as const;

const CHART_COLORS = [
  'hsl(var(--primary))',
  '#ef7d52',
  'hsl(160 84% 39%)',
  'hsl(38 92% 50%)',
  '#8b5cf6',
  '#06b6d4',
];

const AXIS_COLOR = 'hsl(var(--muted-foreground))';
const SUCCESS_COLOR = 'hsl(160 84% 39%)';
const WARNING_COLOR = 'hsl(38 92% 50%)';
const ERROR_COLOR = 'hsl(var(--destructive))';

const LINE_COLORS = [
  '#ef7d52', '#3b82f6', '#22c55e', '#f59e0b',
  '#a855f7', '#ec4899', '#06b6d4', '#84cc16',
];

const formatTokens = (n: number): string => {
  if (!Number.isFinite(n) || n <= 0) return '0';
  if (n >= 1_000_000_000) return `${(n / 1_000_000_000).toFixed(2)}B`;
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return Math.round(n).toLocaleString();
};

const formatAxisTokens = (n: number): string => {
  if (!Number.isFinite(n) || n <= 0) return '0';
  if (n >= 1_000_000_000) return `${(n / 1_000_000_000).toFixed(1)}B`;
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(0)}K`;
  return String(Math.round(n));
};

const formatCost = (n: number) => {
  if (!n || n === 0) return '$0.00';
  if (n >= 1) return `$${n.toFixed(2)}`;
  if (n >= 0.01) return `$${n.toFixed(4)}`;
  return `$${n.toFixed(6)}`;
};

const formatRangeWindow = (startIso?: string, endIso?: string): string => {
  if (!startIso || !endIso) return '';
  try {
    const s = new Date(startIso);
    const e = new Date(endIso);
    const fmt = (d: Date) =>
      `${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')} ${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
    return `${fmt(s)}→${fmt(e)}`;
  } catch {
    return '';
  }
};

const formatClockTime = (isoStr?: string): string => {
  if (!isoStr) return '--:--:--';
  try {
    const s = isoStr.includes('T') ? isoStr : isoStr.replace(' ', 'T') + (isoStr.endsWith('Z') ? '' : 'Z');
    const d = new Date(s);
    if (isNaN(d.getTime())) return isoStr.slice(11, 19) || isoStr;
    return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}:${String(d.getSeconds()).padStart(2, '0')}`;
  } catch {
    return '--:--:--';
  }
};

/** 迷你平滑面积趋势图（SVG Sparkline） */
function MiniSparkline({
  values,
  strokeColor = '#e07a5f',
  gradientId,
}: {
  values: number[];
  strokeColor?: string;
  gradientId: string;
}) {
  const pts = useMemo(() => {
    const clean = values.length > 0 ? values : [0, 0, 0, 0];
    const series = clean.length === 1 ? [clean[0], clean[0]] : clean;
    const w = 240;
    const h = 42;
    const padY = 4;
    const maxV = Math.max(...series, 0.0001);
    const minV = Math.min(...series, 0);
    const span = Math.max(maxV - minV, maxV * 0.15, 0.0001);

    const coords = series.map((v, i) => {
      const x = (i / Math.max(1, series.length - 1)) * w;
      const y = h - padY - ((v - minV) / span) * (h - padY * 2);
      return [x, Math.max(padY, Math.min(h - padY, y))] as const;
    });

    const linePath = coords
      .map(([x, y], idx) => `${idx === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`)
      .join(' ');
    const areaPath = `${linePath} L${w},${h} L0,${h} Z`;
    return { linePath, areaPath, w, h };
  }, [values]);

  return (
    <svg
      viewBox={`0 0 ${pts.w} ${pts.h}`}
      className="w-full h-10 overflow-visible select-none"
      preserveAspectRatio="none"
    >
      <defs>
        <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={strokeColor} stopOpacity={0.34} />
          <stop offset="100%" stopColor={strokeColor} stopOpacity={0.02} />
        </linearGradient>
      </defs>
      <path d={pts.areaPath} fill={`url(#${gradientId})`} />
      <path
        d={pts.linePath}
        fill="none"
        stroke={strokeColor}
        strokeWidth="1.75"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

/** 请求成功率圆环图 */
function SuccessDonutGauge({
  rate,
  total,
}: {
  rate: number;
  total: number;
}) {
  const clamped = total > 0 ? Math.max(0, Math.min(100, rate)) : 0;
  const size = 90;
  const strokeWidth = 8.5;
  const r = (size - strokeWidth) / 2;
  const c = 2 * Math.PI * r;
  const offset = c - (clamped / 100) * c;
  const ringColor = clamped >= 95 ? '#22c55e' : clamped >= 80 ? '#f59e0b' : '#ef4444';

  return (
    <div className="relative flex items-center justify-center my-auto py-0.5">
      <svg width={size} height={size} className="-rotate-90">
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke="currentColor"
          strokeWidth={strokeWidth}
          className="text-muted/60 dark:text-zinc-800"
        />
        {total > 0 && clamped < 100 && (
          <circle
            cx={size / 2}
            cy={size / 2}
            r={r}
            fill="none"
            stroke="#ef4444"
            strokeOpacity={0.45}
            strokeWidth={strokeWidth}
          />
        )}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke={ringColor}
          strokeWidth={strokeWidth}
          strokeDasharray={c}
          strokeDashoffset={offset}
          strokeLinecap="round"
          className="transition-all duration-700 ease-out"
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
        <span className="text-base font-bold tabular-nums text-foreground leading-none">
          {clamped.toFixed(1)}%
        </span>
        <span className="text-[11px] text-muted-foreground mt-1">成功</span>
      </div>
    </div>
  );
}

export default function Dashboard() {
  const { token } = useAuthStore();

  // 全局控制状态：时间范围、自定义时间、模型筛选、渠道筛选
  const [rangeKey, setRangeKey] = useState<RangeKey>('3d');
  const [customStart, setCustomStart] = useState('');
  const [customEnd, setCustomEnd] = useState('');
  const [showCustomTime, setShowCustomTime] = useState(false);

  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [selectedProviders, setSelectedProviders] = useState<string[]>([]);
  const [selectedKeys, setSelectedKeys] = useState<string[]>([]);
  const [showProviderDrawer, setShowProviderDrawer] = useState(false);
  const [showKeyDrawer, setShowKeyDrawer] = useState(false);
  const [analyticsTargetKey, setAnalyticsTargetKey] = useState<{ key_hash: string; key_label: string } | null>(null);

  // 导航与滚动监听状态
  const [activeAnchor, setActiveAnchor] = useState<string>('section-overview');
  const [isScrolled, setIsScrolled] = useState(false);

  // 时序图模式：token (Token+请求) | cache (缓存/未缓存/输出堆叠) | models (分模型折线) | keys (分Key折线) | requests (成功/失败) | throughput (吞吐/首字)
  const [chartMode, setChartMode] = useState<'token' | 'cache' | 'models' | 'keys' | 'requests' | 'throughput'>('token');
  // 是否展开双图并排视图（左边所选时序 + 右边分模型/缓存堆叠对比）
  const [dualChartCompare, setDualChartCompare] = useState(false);

  // 底部明细表 Tab 与搜索
  const [breakdownTab, setBreakdownTab] = useState<'cross' | 'key' | 'key_model' | 'model' | 'provider'>('cross');
  const [tableSearch, setTableSearch] = useState('');

  // 后端数据状态
  const [loading, setLoading] = useState(true);
  const [summary, setSummary] = useState<OverviewSummary | null>(null);
  const [timeseries, setTimeseries] = useState<TimeseriesPoint[]>([]);
  const [modelTimeseries, setModelTimeseries] = useState<Record<string, any>[]>([]);
  const [keyTimeseries, setKeyTimeseries] = useState<Record<string, any>[]>([]);
  const [topModels, setTopModels] = useState<string[]>([]);
  const [topKeys, setTopKeys] = useState<TopKeyItem[]>([]);
  const [availableModels, setAvailableModels] = useState<AvailableModelItem[]>([]);
  const [availableProviders, setAvailableProviders] = useState<AvailableProviderItem[]>([]);
  const [availableKeys, setAvailableKeys] = useState<AvailableKeyItem[]>([]);
  const [recentRequests, setRecentRequests] = useState<RecentRequestItem[]>([]);
  const [recentTotal, setRecentTotal] = useState(0);
  const [heatmap, setHeatmap] = useState<ActivityHeatmapData | null>(null);
  const [modelBreakdown, setModelBreakdown] = useState<BreakdownRow[]>([]);
  const [providerBreakdown, setProviderBreakdown] = useState<BreakdownRow[]>([]);
  const [keyBreakdown, setKeyBreakdown] = useState<KeyBreakdownRow[]>([]);
  const [crossBreakdown, setCrossBreakdown] = useState<CrossBreakdownRow[]>([]);
  const [keyModelBreakdown, setKeyModelBreakdown] = useState<KeyModelBreakdownRow[]>([]);
  const [stats, setStats] = useState<StatData | null>(null);
  const [hoveredDay, setHoveredDay] = useState<HeatmapDay | null>(null);

  const fetchDashboard = async () => {
    if (!token) return;
    setLoading(true);
    const hours = RANGE_OPTIONS.find(r => r.key === rangeKey)?.hours || 72;
    const tzOffset = new Date().getTimezoneOffset();

    const params = new URLSearchParams();
    params.set('range_key', rangeKey);
    params.set('tz_offset_minutes', String(tzOffset));
    params.set('recent_limit', '80');
    if (customStart) params.set('start_datetime', new Date(customStart).toISOString());
    if (customEnd) params.set('end_datetime', new Date(customEnd).toISOString());
    if (selectedModels.length > 0) params.set('model', selectedModels.join(','));
    if (selectedProviders.length > 0) params.set('provider', selectedProviders.join(','));
    if (selectedKeys.length > 0) params.set('key_hash', selectedKeys.join(','));

    try {
      const [overviewRes, statsRes] = await Promise.all([
        apiFetch(`/v1/stats/dashboard_overview?${params.toString()}`),
        apiFetch(`/v1/stats?hours=${hours}`),
      ]);

      if (overviewRes.ok) {
        const data = await overviewRes.json();
        if (!data.empty) {
          setSummary(data.summary || null);
          setTimeseries(data.timeseries || []);
          setModelTimeseries(data.model_timeseries || []);
          setKeyTimeseries(data.key_timeseries || []);
          setTopModels(data.top_models || []);
          setTopKeys(data.top_keys || []);
          setAvailableModels(data.available_models || []);
          setAvailableProviders(data.available_providers || []);
          setAvailableKeys(data.available_keys || []);
          setRecentRequests(data.recent_requests || []);
          setRecentTotal(data.recent_total || 0);
          setHeatmap(data.activity_heatmap || null);
          setModelBreakdown(data.model_breakdown || []);
          setProviderBreakdown(data.provider_breakdown || []);
          setKeyBreakdown(data.key_breakdown || []);
          setCrossBreakdown(data.cross_breakdown || []);
          setKeyModelBreakdown(data.key_model_breakdown || []);
        }
      }

      if (statsRes.ok) {
        const sData = await statsRes.json();
        setStats(sData.stats || sData);
      }
    } catch (err) {
      console.error('Failed to fetch dashboard data:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDashboard();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token, rangeKey, selectedModels, selectedProviders, selectedKeys, customStart, customEnd]);

  // 监听外层 #app-main-scroll 滚动容器，实现 ScrollSpy 高亮与回到顶部状态感知
  useEffect(() => {
    const scrollEl = document.getElementById('app-main-scroll');
    if (!scrollEl) return;

    const onScroll = () => {
      const st = scrollEl.scrollTop;
      setIsScrolled(st > 80);

      let currentSection: string = NAV_SECTIONS[0].id;
      for (const sec of NAV_SECTIONS) {
        const dom = document.getElementById(sec.id);
        if (dom) {
          const rect = dom.getBoundingClientRect();
          const containerRect = scrollEl.getBoundingClientRect();
          if (rect.top - containerRect.top <= 180) {
            currentSection = sec.id;
          }
        }
      }
      setActiveAnchor(currentSection);
    };

    scrollEl.addEventListener('scroll', onScroll, { passive: true });
    return () => scrollEl.removeEventListener('scroll', onScroll);
  }, []);

  const scrollToSection = (id: string) => {
    setActiveAnchor(id);
    const scrollEl = document.getElementById('app-main-scroll');
    const target = document.getElementById(id);
    if (scrollEl && target) {
      const top =
        target.getBoundingClientRect().top -
        scrollEl.getBoundingClientRect().top +
        scrollEl.scrollTop -
        120;
      scrollEl.scrollTo({ top: Math.max(0, top), behavior: 'smooth' });
    } else if (target) {
      target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

  const scrollToTop = () => {
    const scrollEl = document.getElementById('app-main-scroll');
    if (scrollEl) {
      scrollEl.scrollTo({ top: 0, behavior: 'smooth' });
    } else {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  };

  const toggleModelFilter = (modelName: string) => {
    setSelectedModels(prev =>
      prev.includes(modelName) ? prev.filter(m => m !== modelName) : [...prev, modelName]
    );
  };

  const toggleProviderFilter = (provName: string) => {
    setSelectedProviders(prev =>
      prev.includes(provName) ? prev.filter(p => p !== provName) : [...prev, provName]
    );
  };

  const toggleKeyFilter = (keyHash: string) => {
    if (!keyHash) return;
    setSelectedKeys(prev =>
      prev.includes(keyHash) ? prev.filter(k => k !== keyHash) : [...prev, keyHash]
    );
  };

  const clearAllFilters = () => {
    setSelectedModels([]);
    setSelectedProviders([]);
    setSelectedKeys([]);
    setCustomStart('');
    setCustomEnd('');
  };

  const hasActiveFilter =
    selectedModels.length > 0 ||
    selectedProviders.length > 0 ||
    selectedKeys.length > 0 ||
    Boolean(customStart || customEnd);

  // 原版分布数据与过滤后数据的有机结合
  const channelStats = useMemo(() => {
    const raw = stats?.channel_success_rates || [];
    const filtered =
      selectedProviders.length > 0
        ? raw.filter(c =>
            selectedProviders.some(sp => c.provider === sp || c.provider.startsWith(sp + ':'))
          )
        : raw;
    return filtered.slice().sort((a, b) => b.total_requests - a.total_requests);
  }, [stats, selectedProviders]);

  const formattedChannelStats = useMemo(
    () =>
      channelStats
        .filter(c => c.total_requests >= 2)
        .slice(0, 10)
        .map(item => ({
          name: item.provider,
          success_rate: Math.round(item.success_rate * 1000) / 10,
          requests: item.total_requests,
        })),
    [channelStats]
  );

  const modelPieData = useMemo(() => {
    if (modelBreakdown.length > 0) {
      return modelBreakdown.slice(0, 6).map(m => ({
        model: m.model || 'unknown',
        count: m.requests,
        tokens: m.total_tokens,
      }));
    }
    return (stats?.model_request_counts || []).slice(0, 6);
  }, [modelBreakdown, stats]);

  const formattedEndpointStats = useMemo(
    () =>
      (stats?.endpoint_request_counts || []).slice(0, 5).map(item => ({
        name: item.endpoint.replace('POST ', '').replace('GET ', ''),
        value: item.count,
      })),
    [stats]
  );

  // Sparkline 序列提取
  const tokenSeries = useMemo(() => timeseries.map(p => p.total_tokens), [timeseries]);
  const reqSeries = useMemo(() => timeseries.map(p => p.requests), [timeseries]);
  const speedSeries = useMemo(() => timeseries.map(p => p.speed_tok_s), [timeseries]);
  const cacheRateSeries = useMemo(() => timeseries.map(p => p.cache_hit_rate), [timeseries]);
  const costSeries = useMemo(() => timeseries.map(p => p.cost_cny), [timeseries]);

  // Token 消耗构成比例条
  const tokenSegments = useMemo(() => {
    const total = Math.max(1, summary?.total_tokens || 0);
    const cached = summary?.cached_tokens || 0;
    const uncached = summary?.uncached_tokens || 0;
    const comp = summary?.completion_tokens || 0;
    return {
      cachedPct: Math.min(100, (cached / total) * 100),
      uncachedPct: Math.min(100, (uncached / total) * 100),
      compPct: Math.max(1.5, Math.min(100, (comp / total) * 100)),
    };
  }, [summary]);

  // 工作回溯（近一年 53 周 × 7 天）矩阵构造
  const heatmapGrid = useMemo(() => {
    const days = heatmap?.days || [];
    if (days.length === 0) return { weeks: [], monthHeaders: [], maxReqs: 1 };

    const maxReqs = Math.max(...days.map(d => d.requests), 1);
    const firstDate = new Date(days[0].date + 'T00:00:00');
    const firstWeekday = (firstDate.getDay() + 6) % 7;
    const padded: (HeatmapDay | null)[] = Array(firstWeekday).fill(null).concat(days);
    const weeks: (HeatmapDay | null)[][] = [];
    for (let i = 0; i < padded.length; i += 7) {
      const col = padded.slice(i, i + 7);
      while (col.length < 7) col.push(null);
      weeks.push(col);
    }

    const monthHeaders: { colIndex: number; label: string }[] = [];
    let lastMonth = -1;
    weeks.forEach((col, wIdx) => {
      const firstValid = col.find(c => c !== null);
      if (firstValid) {
        const m = Number(firstValid.date.slice(5, 7));
        if (m !== lastMonth) {
          if (monthHeaders.length === 0 || wIdx - monthHeaders[monthHeaders.length - 1].colIndex >= 3) {
            monthHeaders.push({ colIndex: wIdx, label: `${m}月` });
          }
          lastMonth = m;
        }
      }
    });

    return { weeks, monthHeaders, maxReqs };
  }, [heatmap]);

  const getHeatmapCellColor = (reqs: number, maxReqs: number): string => {
    if (reqs <= 0) return 'bg-muted/70 dark:bg-[#2a2927]';
    const ratio = reqs / maxReqs;
    if (ratio < 0.15) return 'bg-[#6b4c3e]';
    if (ratio < 0.4) return 'bg-[#9a5b42]';
    if (ratio < 0.7) return 'bg-[#c96a48]';
    return 'bg-[#ef7d52] shadow-[0_0_6px_rgba(239,125,82,0.45)]';
  };

  // 建立渠道快速检索映射（在渠道健康表上附加缓存率、Token、速度、成本）
  const providerExtraMap = useMemo(() => {
    const map = new Map<string, BreakdownRow>();
    providerBreakdown.forEach(p => {
      if (p.provider) map.set(p.provider, p);
    });
    return map;
  }, [providerBreakdown]);

  // 搜索过滤后的底部明细表数据
  const filteredCrossRows = useMemo(() => {
    const q = tableSearch.trim().toLowerCase();
    if (!q) return crossBreakdown;
    return crossBreakdown.filter(
      r => r.provider.toLowerCase().includes(q) || r.model.toLowerCase().includes(q)
    );
  }, [crossBreakdown, tableSearch]);

  const filteredModelRows = useMemo(() => {
    const q = tableSearch.trim().toLowerCase();
    if (!q) return modelBreakdown;
    return modelBreakdown.filter(r => (r.model || '').toLowerCase().includes(q));
  }, [modelBreakdown, tableSearch]);

  const filteredProviderRows = useMemo(() => {
    const q = tableSearch.trim().toLowerCase();
    if (!q) return providerBreakdown;
    return providerBreakdown.filter(r => (r.provider || '').toLowerCase().includes(q));
  }, [providerBreakdown, tableSearch]);

  const filteredKeyRows = useMemo(() => {
    const q = tableSearch.trim().toLowerCase();
    if (!q) return keyBreakdown;
    return keyBreakdown.filter(
      r =>
        r.key_label.toLowerCase().includes(q) ||
        r.api_key_prefix.toLowerCase().includes(q) ||
        (r.api_key_group || '').toLowerCase().includes(q) ||
        (r.parent_name || '').toLowerCase().includes(q) ||
        (r.top_model || '').toLowerCase().includes(q)
    );
  }, [keyBreakdown, tableSearch]);

  const filteredKeyModelRows = useMemo(() => {
    const q = tableSearch.trim().toLowerCase();
    if (!q) return keyModelBreakdown;
    return keyModelBreakdown.filter(
      r =>
        r.key_label.toLowerCase().includes(q) ||
        r.api_key_prefix.toLowerCase().includes(q) ||
        (r.api_key_group || '').toLowerCase().includes(q) ||
        r.model.toLowerCase().includes(q)
    );
  }, [keyModelBreakdown, tableSearch]);

  const tooltipStyle = {
    backgroundColor: 'hsl(var(--popover))',
    borderColor: 'hsl(var(--border))',
    color: 'hsl(var(--popover-foreground))',
    borderRadius: '10px',
    fontSize: '12px',
    boxShadow: '0 10px 25px -5px rgba(0, 0, 0, 0.4)',
  };

  if (loading && !summary && !stats) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-muted-foreground py-24">
        <RefreshCw className="w-8 h-8 animate-spin mb-3 text-primary" />
        <p>正在加载网关遥测看板...</p>
      </div>
    );
  }

  return (
    <div className="space-y-4 sm:space-y-5 animate-in fade-in duration-300 font-sans pb-14 min-w-0 w-full max-w-full overflow-x-clip">
      {/* =====================================================================
          1. 常驻吸顶全局指挥栏（Sticky Command Bar）
             - 随页面滚动常驻顶部，使「模块跳转」「时间/模型切换」「回到顶部」真正随时可用
             - 集成全局模型与渠道过滤，一键驱动全页所有卡片（尤其是缓存命中率）与图表
         ===================================================================== */}
      <div className="sticky -top-2.5 sm:-top-4 md:-top-8 z-30 pt-1.5 pb-1.5 bg-background/90 backdrop-blur-md border-b border-border/70 shadow-sm transition-all min-w-0">
        <div className="bg-card/95 border border-border/80 rounded-2xl px-2.5 sm:px-3.5 py-2 sm:py-2.5 shadow-sm space-y-2 min-w-0">
          {/* 第一行：锚点跳转 + 时间粒度 + 刷新 & 回到顶部 */}
          <div className="flex flex-col sm:flex-row sm:flex-wrap sm:items-center sm:justify-between gap-2 min-w-0">
            {/* 模块快捷跳转胶囊（移动端单行横向滑动，桌面端自动换行） */}
            <div className="flex items-center gap-1 overflow-x-auto no-scrollbar max-w-full sm:flex-wrap pb-0.5 sm:pb-0">
              {NAV_SECTIONS.map(nav => {
                const isActive = activeAnchor === nav.id;
                return (
                  <button
                    key={nav.id}
                    type="button"
                    onClick={() => scrollToSection(nav.id)}
                    className={`px-2.5 py-1 text-xs font-medium rounded-full transition-all whitespace-nowrap shrink-0 ${
                      isActive
                        ? 'bg-primary/15 text-primary border border-primary/40 font-semibold'
                        : 'text-muted-foreground hover:text-foreground hover:bg-muted/60 border border-transparent'
                    }`}
                  >
                    {nav.label}
                  </button>
                );
              })}
            </div>

            {/* 时间范围 + 操作按钮 */}
            <div className="flex items-center justify-between sm:justify-end gap-1.5 min-w-0">
              <div className="inline-flex items-center bg-muted/40 border border-border/80 rounded-xl p-0.5 gap-0.5 overflow-x-auto no-scrollbar min-w-0 flex-1 sm:flex-initial">
                {RANGE_OPTIONS.map(opt => {
                  const active = rangeKey === opt.key && !customStart && !customEnd;
                  return (
                    <button
                      key={opt.key}
                      type="button"
                      onClick={() => {
                        setCustomStart('');
                        setCustomEnd('');
                        setRangeKey(opt.key);
                      }}
                      className={`px-2 sm:px-2.5 py-1 text-xs font-medium rounded-lg transition-all whitespace-nowrap shrink-0 ${
                        active
                          ? 'bg-primary text-primary-foreground shadow-sm'
                          : 'text-muted-foreground hover:text-foreground hover:bg-muted/60'
                      }`}
                    >
                      {opt.label}
                    </button>
                  );
                })}
                <button
                  type="button"
                  onClick={() => setShowCustomTime(v => !v)}
                  className={`px-2 sm:px-2.5 py-1 text-xs font-medium rounded-lg transition-all inline-flex items-center gap-1 whitespace-nowrap shrink-0 ${
                    customStart || customEnd || showCustomTime
                      ? 'bg-primary/15 text-primary border border-primary/30'
                      : 'text-muted-foreground hover:text-foreground'
                  }`}
                  title="自定义起止时间"
                >
                  <Calendar className="w-3 h-3" />
                  自定义
                </button>
              </div>

              <div className="flex items-center gap-1 shrink-0">
                <button
                  type="button"
                  onClick={fetchDashboard}
                  title="刷新实时数据"
                  className="p-1.5 text-muted-foreground hover:text-foreground bg-muted/40 hover:bg-muted/80 border border-border/80 rounded-xl transition-colors"
                >
                  <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin text-primary' : ''}`} />
                </button>

                <button
                  type="button"
                  onClick={scrollToTop}
                  title="回到顶部"
                  className={`inline-flex items-center gap-1 px-2 sm:px-2.5 py-1 text-xs font-medium rounded-xl border transition-all whitespace-nowrap ${
                    isScrolled
                      ? 'bg-primary text-primary-foreground border-primary shadow-sm'
                      : 'text-muted-foreground hover:text-foreground bg-muted/40 border-border/80'
                  }`}
                >
                  <ArrowUp className="w-3.5 h-3.5" />
                  <span className="hidden sm:inline">回到顶部</span>
                </button>
              </div>
            </div>
          </div>

          {/* 第二行：全局「模型」快捷胶囊筛选（移动端单行横向滑动不遮挡屏幕） + 「Key/渠道」筛选 */}
          <div className="flex flex-col sm:flex-row sm:flex-wrap sm:items-center sm:justify-between gap-1.5 sm:gap-2 pt-1.5 border-t border-border/50 min-w-0">
            <div className="flex items-center gap-1.5 overflow-x-auto no-scrollbar max-w-full sm:flex-wrap sm:flex-1 min-w-0 pb-0.5 sm:pb-0">
              <span className="inline-flex items-center gap-1 text-[11px] font-medium text-muted-foreground shrink-0 mr-0.5">
                <Filter className="w-3 h-3 text-primary" />
                模型:
              </span>
              <button
                type="button"
                onClick={() => setSelectedModels([])}
                className={`px-2.5 py-0.5 text-[11px] font-medium rounded-lg border transition-all whitespace-nowrap shrink-0 ${
                  selectedModels.length === 0
                    ? 'bg-foreground text-background border-foreground font-semibold'
                    : 'bg-muted/40 text-muted-foreground border-border/70 hover:text-foreground'
                }`}
              >
                全部模型 ({availableModels.length})
              </button>

              {availableModels.slice(0, 8).map(item => {
                const isSelected = selectedModels.includes(item.model);
                return (
                  <button
                    key={item.model}
                    type="button"
                    onClick={() => toggleModelFilter(item.model)}
                    title={`点击只看 ${item.model}（请求 ${item.requests} 次 · 缓存命中率 ${item.cache_hit_rate}% · 速度 ${item.speed_tok_s} tok/s）`}
                    className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 text-[11px] rounded-lg border transition-all tabular-nums whitespace-nowrap shrink-0 ${
                      isSelected
                        ? 'bg-primary text-primary-foreground border-primary font-semibold shadow-sm'
                        : 'bg-muted/30 text-foreground/85 border-border/70 hover:bg-muted/70'
                    }`}
                  >
                    <span className="truncate max-w-[135px] sm:max-w-[150px]">{item.model}</span>
                    <span
                      className={`text-[10px] px-1 rounded shrink-0 ${
                        isSelected
                          ? 'bg-primary-foreground/20 text-primary-foreground'
                          : item.cache_hit_rate >= 50
                          ? 'bg-cyan-500/15 text-cyan-400'
                          : 'bg-muted text-muted-foreground'
                      }`}
                    >
                      {item.cache_hit_rate.toFixed(0)}%♻
                    </span>
                  </button>
                );
              })}
            </div>

            {/* 右侧：Key 筛选开关 + 渠道筛选开关 + 一键重置过滤 */}
            <div className="flex items-center gap-1.5 overflow-x-auto no-scrollbar max-w-full shrink-0 sm:flex-wrap">
              <button
                type="button"
                onClick={() => setShowKeyDrawer(v => !v)}
                className={`inline-flex items-center gap-1 px-2.5 py-0.5 text-[11px] font-medium rounded-lg border transition-colors whitespace-nowrap shrink-0 ${
                  selectedKeys.length > 0 || showKeyDrawer
                    ? 'bg-amber-500/15 text-amber-400 border-amber-500/40'
                    : 'bg-muted/40 text-muted-foreground border-border/70 hover:text-foreground'
                }`}
              >
                <KeyRound className="w-3 h-3" />
                {selectedKeys.length > 0
                  ? `已选 ${selectedKeys.length} 条 Key`
                  : `按 Key 筛选 (${availableKeys.length})`}
              </button>

              <button
                type="button"
                onClick={() => setShowProviderDrawer(v => !v)}
                className={`inline-flex items-center gap-1 px-2.5 py-0.5 text-[11px] font-medium rounded-lg border transition-colors whitespace-nowrap shrink-0 ${
                  selectedProviders.length > 0 || showProviderDrawer
                    ? 'bg-primary/15 text-primary border-primary/40'
                    : 'bg-muted/40 text-muted-foreground border-border/70 hover:text-foreground'
                }`}
              >
                <Server className="w-3 h-3" />
                {selectedProviders.length > 0
                  ? `已选 ${selectedProviders.length} 个渠道`
                  : `按渠道筛选 (${availableProviders.length})`}
              </button>

              {hasActiveFilter && (
                <button
                  type="button"
                  onClick={clearAllFilters}
                  className="inline-flex items-center gap-1 px-2 py-0.5 text-[11px] font-medium text-rose-400 hover:bg-rose-500/10 border border-rose-500/30 rounded-lg transition-colors whitespace-nowrap shrink-0"
                >
                  <X className="w-3 h-3" />
                  清除筛选
                </button>
              )}
            </div>
          </div>

          {/* 展开的 API Key 多选抽屉 */}
          {showKeyDrawer && (
            <div className="pt-2 border-t border-border/50 flex flex-wrap items-center gap-1.5 max-h-36 overflow-y-auto">
              <span className="text-[11px] text-muted-foreground mr-1">选择 API Key:</span>
              <button
                type="button"
                onClick={() => setSelectedKeys([])}
                className={`px-2 py-0.5 text-[11px] rounded-md border ${
                  selectedKeys.length === 0
                    ? 'bg-foreground text-background border-foreground font-semibold'
                    : 'bg-muted/40 text-muted-foreground border-border/70'
                }`}
              >
                全部 Key ({availableKeys.length})
              </button>
              {availableKeys.map(k => {
                const active = selectedKeys.includes(k.key_hash);
                return (
                  <button
                    key={k.key_hash}
                    type="button"
                    onClick={() => toggleKeyFilter(k.key_hash)}
                    title={`${k.key_label} (${k.api_key_prefix})${k.api_key_group ? ` · 分组 ${k.api_key_group}` : ''} · 请求 ${k.requests} 次 · Token ${formatTokens(k.total_tokens)} · ¥${k.cost_cny.toFixed(2)}`}
                    className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 text-[11px] rounded-md border transition-colors tabular-nums ${
                      active
                        ? 'bg-amber-500 text-slate-950 border-amber-500 font-semibold shadow-sm'
                        : 'bg-muted/30 text-muted-foreground border-border/70 hover:text-foreground'
                    }`}
                  >
                    <KeyRound className="w-2.5 h-2.5 opacity-80 shrink-0" />
                    <span className="truncate max-w-[120px] sm:max-w-[140px]">{k.key_label}</span>
                    {k.api_key_group && (
                      <span
                        className={`text-[9px] px-1 rounded shrink-0 ${
                          active ? 'bg-slate-950/20 text-slate-950' : 'bg-muted text-muted-foreground'
                        }`}
                      >
                        {k.api_key_group}
                      </span>
                    )}
                    <span className="opacity-75 text-[10px] shrink-0">({formatTokens(k.total_tokens)})</span>
                  </button>
                );
              })}
            </div>
          )}

          {/* 展开的渠道多选抽屉 */}
          {showProviderDrawer && (
            <div className="pt-2 border-t border-border/50 flex flex-wrap items-center gap-1.5 max-h-36 overflow-y-auto">
              <span className="text-[11px] text-muted-foreground mr-1">选择渠道:</span>
              <button
                type="button"
                onClick={() => setSelectedProviders([])}
                className={`px-2 py-0.5 text-[11px] rounded-md border ${
                  selectedProviders.length === 0
                    ? 'bg-foreground text-background border-foreground font-semibold'
                    : 'bg-muted/40 text-muted-foreground border-border/70'
                }`}
              >
                全部渠道
              </button>
              {availableProviders.map(p => {
                const active = selectedProviders.includes(p.provider);
                return (
                  <button
                    key={p.provider}
                    type="button"
                    onClick={() => toggleProviderFilter(p.provider)}
                    className={`inline-flex items-center gap-1 px-2 py-0.5 text-[11px] rounded-md border transition-colors tabular-nums ${
                      active
                        ? 'bg-primary text-primary-foreground border-primary font-semibold'
                        : 'bg-muted/30 text-muted-foreground border-border/70 hover:text-foreground'
                    }`}
                  >
                    <span>{p.provider}</span>
                    <span className="opacity-70 text-[10px]">({p.requests})</span>
                  </button>
                );
              })}
            </div>
          )}

          {/* 展开的自定义起止时间栏 */}
          {showCustomTime && (
            <div className="pt-2 border-t border-border/50 flex flex-col sm:flex-row sm:flex-wrap sm:items-center gap-2 text-xs">
              <span className="text-muted-foreground">起止时间:</span>
              <input
                type="datetime-local"
                value={customStart}
                onChange={e => setCustomStart(e.target.value)}
                className="px-2.5 py-1 text-xs bg-background border border-border rounded-lg text-foreground w-full sm:w-auto"
              />
              <span className="text-muted-foreground hidden sm:inline">→</span>
              <input
                type="datetime-local"
                value={customEnd}
                onChange={e => setCustomEnd(e.target.value)}
                className="px-2.5 py-1 text-xs bg-background border border-border rounded-lg text-foreground w-full sm:w-auto"
              />
              {(customStart || customEnd) && (
                <button
                  type="button"
                  onClick={() => {
                    setCustomStart('');
                    setCustomEnd('');
                  }}
                  className="text-xs text-rose-400 hover:underline self-start sm:self-auto"
                >
                  清除时间
                </button>
              )}
            </div>
          )}
        </div>
      </div>

      {/* =====================================================================
          2. 核心 6 指标卡片矩阵 + 主时序图 & 实时活动流
         ===================================================================== */}
      <div id="section-overview" className="space-y-4 min-w-0">
        <div className="grid grid-cols-2 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-2.5 sm:gap-3.5 min-w-0">
          {/* Card 1: TOKEN 消耗 */}
          <div className="relative bg-card border border-border/80 rounded-2xl p-3 sm:p-4 shadow-sm overflow-hidden flex flex-col justify-between min-h-[172px] sm:min-h-[212px] min-w-0">
            <div className="absolute left-0 top-3 bottom-3 w-1 rounded-r-full bg-gradient-to-b from-indigo-500 via-purple-500 to-amber-500" />
            <div className="min-w-0">
              <div className="flex items-center justify-between gap-1 text-[11px] sm:text-xs font-medium text-muted-foreground">
                <span className="inline-flex items-center gap-1 sm:gap-1.5 truncate">
                  <span className="text-indigo-400">◆</span>
                  <span className="truncate">TOKEN 消耗</span>
                </span>
                {selectedModels.length === 1 && (
                  <span className="text-[10px] text-primary truncate max-w-[65px] sm:max-w-[90px]" title={selectedModels[0]}>
                    {selectedModels[0]}
                  </span>
                )}
              </div>
              <div className="mt-1.5 sm:mt-2 text-2xl sm:text-3xl font-bold tracking-tight tabular-nums text-foreground truncate">
                {formatTokens(summary?.total_tokens || 0)}
              </div>
              <div className="mt-1.5 sm:mt-2 space-y-0.5 sm:space-y-1 text-[10px] sm:text-[11px] text-muted-foreground tabular-nums leading-snug">
                <div className="flex flex-wrap items-center gap-x-1">
                  <span>输入 <strong className="font-semibold text-foreground/90">{formatTokens(summary?.prompt_tokens || 0)}</strong></span>
                  <span>·</span>
                  <span>输出 <strong className="font-semibold text-foreground/90">{formatTokens(summary?.completion_tokens || 0)}</strong></span>
                </div>
                <div className="flex flex-wrap items-center gap-x-1">
                  <span>未命中 <strong className="font-semibold text-foreground/90">{formatTokens(summary?.uncached_tokens || 0)}</strong></span>
                  <span>·</span>
                  <span>缓存 <strong className="font-semibold text-foreground/90">{formatTokens(summary?.cached_tokens || 0)}</strong></span>
                </div>
              </div>
              <div className="mt-2 sm:mt-2.5 flex items-center gap-1">
                <div className="flex-1 h-1.5 rounded-full bg-muted overflow-hidden flex">
                  <div
                    className="h-full bg-slate-300 dark:bg-slate-200 transition-all duration-500"
                    style={{ width: `${tokenSegments.cachedPct}%` }}
                    title={`缓存命中: ${tokenSegments.cachedPct.toFixed(1)}%`}
                  />
                  <div
                    className="h-full bg-amber-500 transition-all duration-500"
                    style={{ width: `${tokenSegments.uncachedPct}%` }}
                    title={`未缓存输入: ${tokenSegments.uncachedPct.toFixed(1)}%`}
                  />
                  <div
                    className="h-full bg-emerald-500 transition-all duration-500"
                    style={{ width: `${tokenSegments.compPct}%` }}
                    title={`输出 Token: ${tokenSegments.compPct.toFixed(1)}%`}
                  />
                </div>
                <span className="w-1.5 h-1.5 rounded-full bg-amber-400 shrink-0" />
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 shrink-0" />
              </div>
            </div>
            <div className="mt-2 sm:mt-2.5 pt-1">
              <MiniSparkline values={tokenSeries} strokeColor="#e07a5f" gradientId="spark-token" />
            </div>
          </div>

          {/* Card 2: 请求数 */}
          <div className="bg-card border border-border/80 rounded-2xl p-3 sm:p-4 shadow-sm overflow-hidden flex flex-col justify-between min-h-[172px] sm:min-h-[212px] min-w-0">
            <div className="min-w-0">
              <div className="flex flex-wrap items-center justify-between gap-1 text-[11px] sm:text-xs font-medium text-muted-foreground">
                <span className="inline-flex items-center gap-1 sm:gap-1.5 shrink-0">
                  <span className="text-sky-400">⇄</span>
                  <span>请求数</span>
                </span>
                <div className="inline-flex items-center gap-1 flex-wrap">
                  <button
                    type="button"
                    onClick={() => scrollToSection('section-key-usage')}
                    title="点击跳转到 Key 用量排行"
                    className="inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded-md bg-amber-500/10 text-amber-400 hover:bg-amber-500/20 text-[10px] transition-colors whitespace-nowrap"
                  >
                    Key {summary?.active_keys ?? keyBreakdown.length}
                  </button>
                  <span className="inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded-md bg-purple-500/10 text-purple-400 text-[10px] whitespace-nowrap">
                    渠道 {providerBreakdown.length}
                  </span>
                </div>
              </div>
              <div className="mt-1.5 sm:mt-2 flex items-baseline gap-1 min-w-0">
                <span className="text-2xl sm:text-3xl font-bold tracking-tight tabular-nums text-foreground truncate">
                  {(summary?.total_requests || 0).toLocaleString()}
                </span>
                <span className="text-xs text-muted-foreground shrink-0">次</span>
              </div>
              <div className="mt-1.5 sm:mt-2 text-[10px] sm:text-[11px] text-muted-foreground tabular-nums leading-snug flex flex-wrap items-center gap-x-1">
                <span>失败 <strong className="font-semibold text-foreground/90">{(summary?.failed_requests || 0).toLocaleString()}</strong></span>
                <span>·</span>
                <span>成功率 <strong className="font-semibold text-foreground/90">{(summary?.success_rate || 0).toFixed(1)}%</strong></span>
              </div>
            </div>
            <div className="mt-2 sm:mt-2.5 pt-1">
              <MiniSparkline values={reqSeries} strokeColor="#6488ea" gradientId="spark-req" />
            </div>
          </div>

          {/* Card 3: 生成速度 */}
          <div className="bg-card border border-border/80 rounded-2xl p-3 sm:p-4 shadow-sm overflow-hidden flex flex-col justify-between min-h-[172px] sm:min-h-[212px] min-w-0">
            <div className="min-w-0">
              <div className="flex items-center gap-1 sm:gap-1.5 text-[11px] sm:text-xs font-medium text-muted-foreground">
                <span className="text-amber-400">⚡</span>
                <span className="truncate">生成速度</span>
              </div>
              <div className="mt-1.5 sm:mt-2 flex items-baseline gap-1 min-w-0">
                <span className="text-2xl sm:text-3xl font-bold tracking-tight tabular-nums text-foreground truncate">
                  {(summary?.avg_speed_tok_s || 0).toFixed(1)}
                </span>
                <span className="text-xs text-muted-foreground shrink-0">tok/s</span>
              </div>
              <div className="mt-1.5 sm:mt-2 space-y-0.5 sm:space-y-1 text-[10px] sm:text-[11px] text-muted-foreground tabular-nums leading-snug">
                <div className="flex flex-wrap items-center gap-x-1">
                  <span>耗时 <strong className="font-semibold text-foreground/90">{(summary?.avg_process_time || 0).toFixed(2)}s</strong></span>
                  <span>·</span>
                  <span>首字 <strong className="font-semibold text-foreground/90">{(summary?.avg_first_response_time || 0).toFixed(2)}s</strong></span>
                </div>
                <div className="flex flex-wrap items-center gap-x-1">
                  <span>峰值 <strong className="font-semibold text-foreground/90">{Math.round(summary?.peak_speed_tok_s || 0)}</strong></span>
                  <span>·</span>
                  <span>p90 <strong className="font-semibold text-foreground/90">{(summary?.p90_first_response_time || 0).toFixed(2)}s</strong></span>
                </div>
              </div>
            </div>
            <div className="mt-2 sm:mt-2.5 pt-1">
              <MiniSparkline values={speedSeries} strokeColor="#e07a5f" gradientId="spark-speed" />
            </div>
          </div>

          {/* Card 4: 缓存命中率（支持单模型精准展示 & 多模型独立命中率迷你排行） */}
          <div className="bg-card border border-border/80 rounded-2xl p-3 sm:p-4 shadow-sm overflow-hidden flex flex-col justify-between min-h-[172px] sm:min-h-[212px] min-w-0">
            <div className="min-w-0">
              <div className="flex flex-wrap items-center justify-between gap-1 text-[11px] sm:text-xs font-medium text-muted-foreground">
                <span className="inline-flex items-center gap-1 sm:gap-1.5 shrink-0">
                  <span className="text-cyan-400">♻</span>
                  <span>缓存命中率</span>
                </span>
                <span className="text-[10px] text-cyan-400 font-medium hidden sm:inline">
                  {selectedModels.length === 1
                    ? '单模型精确'
                    : selectedModels.length > 1
                    ? `已选 ${selectedModels.length} 模型`
                    : '点击模型可单看'}
                </span>
              </div>
              <div className="mt-1.5 sm:mt-2 text-2xl sm:text-3xl font-bold tracking-tight tabular-nums text-foreground truncate">
                {(summary?.cache_hit_rate || 0).toFixed(1)}%
              </div>
              <div className="mt-1 sm:mt-1.5 text-[10px] sm:text-[11px] text-muted-foreground tabular-nums leading-snug">
                缓存复用 <span className="font-semibold text-foreground/90">{formatTokens(summary?.cached_tokens || 0)}</span> tok
              </div>

              {/* 当处于「全部模型」时，直接列出 Top 模型的各自独立缓存命中率，点击任意一行立刻只看该模型 */}
              {selectedModels.length === 0 && modelBreakdown.length > 0 && (
                <div className="mt-1.5 sm:mt-2 space-y-1 border-t border-border/50 pt-1.5">
                  {modelBreakdown.slice(0, 2).map(m => (
                    <button
                      key={m.model}
                      type="button"
                      onClick={() => m.model && toggleModelFilter(m.model)}
                      className="w-full flex items-center justify-between gap-1 text-[10px] text-muted-foreground hover:text-foreground transition-colors tabular-nums min-w-0"
                      title={`点击只看 ${m.model} 的缓存与速度`}
                    >
                      <span className="truncate flex-1 text-left">{m.model}</span>
                      <span className="font-semibold text-cyan-400 shrink-0">{m.cache_hit_rate.toFixed(1)}%</span>
                    </button>
                  ))}
                </div>
              )}
            </div>
            <div className="mt-2 pt-0.5">
              <MiniSparkline values={cacheRateSeries} strokeColor="#6488ea" gradientId="spark-cache" />
            </div>
          </div>

          {/* Card 5: API 等价 / 费用 */}
          <div className="bg-card border border-border/80 rounded-2xl p-3 sm:p-4 shadow-sm overflow-hidden flex flex-col justify-between min-h-[172px] sm:min-h-[212px] min-w-0">
            <div className="min-w-0">
              <div className="flex items-center gap-1 sm:gap-1.5 text-[11px] sm:text-xs font-medium text-muted-foreground">
                <span className="text-rose-400">¥</span>
                <span className="truncate">API 等价 / 费用</span>
              </div>
              <div className="mt-1.5 sm:mt-2 text-2xl sm:text-3xl font-bold tracking-tight tabular-nums text-rose-400 truncate">
                ¥ {(summary?.total_cost_cny || 0).toFixed(2)}
              </div>
              <div className="mt-1.5 sm:mt-2 space-y-0.5 sm:space-y-1 text-[10px] sm:text-[11px] text-muted-foreground tabular-nums leading-snug">
                <div className="flex flex-wrap items-center gap-x-1">
                  <span>折合 <strong className="text-emerald-400 font-semibold">${(summary?.total_cost_usd || 0).toFixed(2)}</strong></span>
                  <span>·</span>
                  <span>实记 <strong className="font-semibold text-foreground/90">${(summary?.recorded_cost_usd || 0).toFixed(2)}</strong></span>
                </div>
                <div>
                  单次均价 ¥
                  {summary && summary.total_requests > 0
                    ? (summary.total_cost_cny / summary.total_requests).toFixed(3)
                    : '0.000'}
                </div>
              </div>
            </div>
            <div className="mt-2 sm:mt-2.5 pt-1">
              <MiniSparkline values={costSeries} strokeColor="#e07a5f" gradientId="spark-cost" />
            </div>
          </div>

          {/* Card 6: 请求成功率 */}
          <div className="bg-card border border-border/80 rounded-2xl p-3 sm:p-4 shadow-sm overflow-hidden flex flex-col justify-between min-h-[172px] sm:min-h-[212px] min-w-0">
            <div className="flex items-center gap-1 sm:gap-1.5 text-[11px] sm:text-xs font-medium text-muted-foreground">
              <span className="text-emerald-400">✓</span>
              <span className="truncate">请求成功率</span>
            </div>
            <SuccessDonutGauge
              rate={summary?.success_rate || 0}
              total={summary?.total_requests || 0}
            />
            <div className="text-center text-[10px] sm:text-[11px] text-muted-foreground tabular-nums truncate">
              {(summary?.total_requests || 0).toLocaleString()} 次 · 失败{' '}
              <span className="font-semibold text-foreground/90">{(summary?.failed_requests || 0).toLocaleString()}</span> 次
            </div>
          </div>
        </div>

        {/* 中部双栏：五合一主时序面板 + 最近请求实时活动流 */}
        <div className="grid grid-cols-1 xl:grid-cols-12 gap-4 min-w-0">
          {/* 左栏：统一多模式时序分析面板（消除原版上下两套重复图表） */}
          <div
            className={`${
              dualChartCompare ? 'xl:col-span-12' : 'xl:col-span-7'
            } bg-card border border-border/80 rounded-2xl p-3.5 sm:p-5 shadow-sm flex flex-col justify-between min-w-0 overflow-hidden`}
          >
            <div className="flex flex-col sm:flex-row sm:flex-wrap sm:items-center sm:justify-between gap-2 mb-3 sm:mb-4 min-w-0">
              <div className="flex items-baseline gap-2 flex-wrap min-w-0">
                <h3 className="text-sm font-bold tracking-wide text-foreground uppercase">
                  TOKEN 与请求时序
                </h3>
                <span className="text-[11px] sm:text-xs text-muted-foreground tabular-nums">
                  {formatRangeWindow(summary?.start_datetime, summary?.end_datetime)}
                  {' · '}
                  {summary?.granularity === 'day' ? '日粒度' : '小时粒度'}
                  {' · '}
                  总 {formatTokens(summary?.total_tokens || 0)} tok / {(summary?.total_requests || 0).toLocaleString()} 请求
                </span>
              </div>

              <div className="flex items-center gap-1.5 min-w-0 max-w-full">
                <div className="inline-flex items-center bg-muted/50 border border-border/80 rounded-lg p-0.5 gap-0.5 overflow-x-auto no-scrollbar max-w-full flex-nowrap sm:flex-wrap min-w-0 flex-1 sm:flex-initial">
                  {[
                    { key: 'token', label: 'Token+请求' },
                    { key: 'cache', label: '缓存堆叠' },
                    { key: 'models', label: '分模型趋势' },
                    { key: 'keys', label: '分Key趋势' },
                    { key: 'requests', label: '成功/失败' },
                    { key: 'throughput', label: '吞吐与首字' },
                  ].map(tab => (
                    <button
                      key={tab.key}
                      type="button"
                      onClick={() =>
                        setChartMode(tab.key as 'token' | 'cache' | 'models' | 'keys' | 'requests' | 'throughput')
                      }
                      className={`px-2 sm:px-2.5 py-1 text-xs font-medium rounded-md transition-all whitespace-nowrap shrink-0 ${
                        chartMode === tab.key
                          ? 'bg-card text-foreground shadow-sm border border-border/60 font-semibold'
                          : 'text-muted-foreground hover:text-foreground'
                      }`}
                    >
                      {tab.label}
                    </button>
                  ))}
                </div>

                <button
                  type="button"
                  onClick={() => setDualChartCompare(v => !v)}
                  title="切换单图 / 双图并排对比（同时查看分模型折线与缓存堆叠柱状图）"
                  className={`inline-flex items-center gap-1 px-2 sm:px-2.5 py-1 text-xs font-medium rounded-lg border transition-colors whitespace-nowrap shrink-0 ${
                    dualChartCompare
                      ? 'bg-primary/15 text-primary border-primary/40'
                      : 'bg-muted/40 text-muted-foreground border-border/70 hover:text-foreground'
                  }`}
                >
                  <Layers className="w-3.5 h-3.5" />
                  <span className="hidden sm:inline">双图并排</span>
                </button>
              </div>
            </div>

            {dualChartCompare ? (
              /* 双图并排模式：移动端上下自适应高度堆叠，桌面端左右并排 */
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 sm:gap-5 h-auto lg:h-[280px] min-w-0">
                <div className="bg-muted/20 border border-border/60 rounded-xl p-3 flex flex-col h-[230px] lg:h-full min-w-0">
                  <div className="text-xs font-semibold text-foreground mb-2">
                    分模型请求次数趋势
                  </div>
                  <div className="flex-1 min-h-0">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={modelTimeseries} margin={{ top: 5, right: 5, left: -10, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border) / 0.45)" vertical={false} />
                        <XAxis dataKey="label" stroke={AXIS_COLOR} fontSize={10} tickLine={false} axisLine={false} minTickGap={20} />
                        <YAxis stroke={AXIS_COLOR} fontSize={10} tickLine={false} axisLine={false} />
                        <Tooltip contentStyle={tooltipStyle} labelFormatter={(_, p) => p?.[0]?.payload?.bucket || ''} />
                        <Legend iconType="circle" wrapperStyle={{ fontSize: '11px' }} />
                        {topModels.map((m, i) => (
                          <Line
                            key={m}
                            type="monotone"
                            dataKey={m}
                            name={m}
                            stroke={LINE_COLORS[i % LINE_COLORS.length]}
                            strokeWidth={2}
                            dot={false}
                            connectNulls
                          />
                        ))}
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                <div className="bg-muted/20 border border-border/60 rounded-xl p-3 flex flex-col h-[230px] lg:h-full min-w-0">
                  <div className="text-xs font-semibold text-foreground mb-2">
                    Token 缓存命中 / 未命中 / 输出堆叠
                  </div>
                  <div className="flex-1 min-h-0">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={timeseries} margin={{ top: 5, right: 5, left: -10, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border) / 0.45)" vertical={false} />
                        <XAxis dataKey="label" stroke={AXIS_COLOR} fontSize={10} tickLine={false} axisLine={false} minTickGap={20} />
                        <YAxis stroke={AXIS_COLOR} fontSize={10} tickLine={false} axisLine={false} tickFormatter={formatAxisTokens} />
                        <Tooltip
                          contentStyle={tooltipStyle}
                          labelFormatter={(_, p) => p?.[0]?.payload?.bucket || ''}
                          formatter={(value: number, name: string) => [formatTokens(value), name]}
                        />
                        <Legend wrapperStyle={{ fontSize: '11px' }} />
                        <Bar dataKey="completion_tokens" stackId="tok" name="输出 Token" fill="#10b981" maxBarSize={16} />
                        <Bar dataKey="uncached_tokens" stackId="tok" name="输入(未命中缓存)" fill="#f59e0b" maxBarSize={16} />
                        <Bar dataKey="cached_tokens" stackId="tok" name="输入(命中缓存)" fill="#38bdf8" radius={[4, 4, 0, 0]} maxBarSize={16} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>
            ) : (
              <div className="h-[235px] sm:h-[275px] w-full min-w-0">
                {timeseries.length === 0 ? (
                  <div className="h-full flex items-center justify-center text-xs text-muted-foreground">
                    当前筛选条件下暂无时序记录
                  </div>
                ) : chartMode === 'token' ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={timeseries} margin={{ top: 8, right: 6, left: -8, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border) / 0.45)" vertical={false} />
                      <XAxis dataKey="label" stroke={AXIS_COLOR} fontSize={11} tickLine={false} axisLine={false} minTickGap={24} />
                      <YAxis yAxisId="left" stroke={AXIS_COLOR} fontSize={11} tickLine={false} axisLine={false} tickFormatter={formatAxisTokens} />
                      <YAxis yAxisId="right" orientation="right" hide />
                      <Tooltip
                        contentStyle={tooltipStyle}
                        labelFormatter={(_, payload) => payload?.[0]?.payload?.bucket || ''}
                        formatter={(val: number, name: string) => {
                          if (name === 'Token 总量') return [formatTokens(val), name];
                          return [`${val.toLocaleString()} 次`, name];
                        }}
                      />
                      <Bar yAxisId="left" dataKey="total_tokens" name="Token 总量" fill="#ef7d52" radius={[4, 4, 0, 0]} maxBarSize={16} />
                      <Line yAxisId="right" type="monotone" dataKey="requests" name="请求数" stroke="#7c9cbf" strokeWidth={1.75} dot={false} activeDot={{ r: 4 }} />
                    </ComposedChart>
                  </ResponsiveContainer>
                ) : chartMode === 'cache' ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={timeseries} margin={{ top: 8, right: 6, left: -8, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border) / 0.45)" vertical={false} />
                      <XAxis dataKey="label" stroke={AXIS_COLOR} fontSize={11} tickLine={false} axisLine={false} minTickGap={24} />
                      <YAxis yAxisId="left" stroke={AXIS_COLOR} fontSize={11} tickLine={false} axisLine={false} tickFormatter={formatAxisTokens} />
                      <YAxis yAxisId="right" orientation="right" domain={[0, 100]} hide />
                      <Tooltip
                        contentStyle={tooltipStyle}
                        labelFormatter={(_, payload) => payload?.[0]?.payload?.bucket || ''}
                        formatter={(val: number, name: string) => {
                          if (name.includes('%')) return [`${val.toFixed(1)}%`, name];
                          return [formatTokens(val), name];
                        }}
                      />
                      <Bar yAxisId="left" dataKey="completion_tokens" stackId="tok" name="输出 Token" fill="#10b981" maxBarSize={16} />
                      <Bar yAxisId="left" dataKey="uncached_tokens" stackId="tok" name="输入(未命中缓存)" fill="#f59e0b" maxBarSize={16} />
                      <Bar yAxisId="left" dataKey="cached_tokens" stackId="tok" name="输入(命中缓存)" fill="#38bdf8" radius={[4, 4, 0, 0]} maxBarSize={16} />
                      <Line yAxisId="right" type="monotone" dataKey="cache_hit_rate" name="缓存命中率 (%)" stroke="#e07a5f" strokeWidth={1.75} dot={false} />
                    </ComposedChart>
                  </ResponsiveContainer>
                ) : chartMode === 'models' ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={modelTimeseries} margin={{ top: 8, right: 6, left: -8, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border) / 0.45)" vertical={false} />
                      <XAxis dataKey="label" stroke={AXIS_COLOR} fontSize={11} tickLine={false} axisLine={false} minTickGap={24} />
                      <YAxis stroke={AXIS_COLOR} fontSize={11} tickLine={false} axisLine={false} />
                      <Tooltip contentStyle={tooltipStyle} labelFormatter={(_, p) => p?.[0]?.payload?.bucket || ''} />
                      <Legend iconType="circle" wrapperStyle={{ fontSize: '11px', paddingTop: '6px' }} />
                      {topModels.map((m, i) => (
                        <Line
                          key={m}
                          type="monotone"
                          dataKey={m}
                          name={m}
                          stroke={LINE_COLORS[i % LINE_COLORS.length]}
                          strokeWidth={2}
                          dot={false}
                          connectNulls
                          activeDot={{ r: 4 }}
                        />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                ) : chartMode === 'keys' ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={keyTimeseries} margin={{ top: 8, right: 6, left: -8, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border) / 0.45)" vertical={false} />
                      <XAxis dataKey="label" stroke={AXIS_COLOR} fontSize={11} tickLine={false} axisLine={false} minTickGap={24} />
                      <YAxis stroke={AXIS_COLOR} fontSize={11} tickLine={false} axisLine={false} tickFormatter={formatAxisTokens} />
                      <Tooltip
                        contentStyle={tooltipStyle}
                        labelFormatter={(_, p) => p?.[0]?.payload?.bucket || ''}
                        formatter={(val: number, name: string) => [formatTokens(val), name]}
                      />
                      <Legend iconType="circle" wrapperStyle={{ fontSize: '11px', paddingTop: '6px' }} />
                      {topKeys.map((tk, i) => (
                        <Line
                          key={tk.key_hash}
                          type="monotone"
                          dataKey={`${tk.key_hash}__tokens`}
                          name={tk.key_label}
                          stroke={LINE_COLORS[i % LINE_COLORS.length]}
                          strokeWidth={2}
                          dot={false}
                          connectNulls
                          activeDot={{ r: 4 }}
                        />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                ) : chartMode === 'requests' ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={timeseries} margin={{ top: 8, right: 6, left: -8, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border) / 0.45)" vertical={false} />
                      <XAxis dataKey="label" stroke={AXIS_COLOR} fontSize={11} tickLine={false} axisLine={false} minTickGap={24} />
                      <YAxis yAxisId="left" stroke={AXIS_COLOR} fontSize={11} tickLine={false} axisLine={false} />
                      <YAxis yAxisId="right" orientation="right" domain={[0, 100]} hide />
                      <Tooltip contentStyle={tooltipStyle} />
                      <Bar yAxisId="left" dataKey="success_requests" stackId="req" name="成功请求" fill="#22c55e" maxBarSize={16} />
                      <Bar yAxisId="left" dataKey="failed_requests" stackId="req" name="失败请求" fill="#ef4444" radius={[4, 4, 0, 0]} maxBarSize={16} />
                      <Line yAxisId="right" type="monotone" dataKey="success_rate" name="成功率 (%)" stroke="#60a5fa" strokeWidth={1.75} dot={false} />
                    </ComposedChart>
                  </ResponsiveContainer>
                ) : (
                  <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={timeseries} margin={{ top: 8, right: 6, left: -8, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border) / 0.45)" vertical={false} />
                      <XAxis dataKey="label" stroke={AXIS_COLOR} fontSize={11} tickLine={false} axisLine={false} minTickGap={24} />
                      <YAxis yAxisId="left" stroke={AXIS_COLOR} fontSize={11} tickLine={false} axisLine={false} tickFormatter={(v: number) => `${v} t/s`} />
                      <YAxis yAxisId="right" orientation="right" hide />
                      <Tooltip contentStyle={tooltipStyle} />
                      <Bar yAxisId="left" dataKey="speed_tok_s" name="生成速度 (tok/s)" fill="#f59e0b" radius={[4, 4, 0, 0]} maxBarSize={16} />
                      <Line yAxisId="right" type="monotone" dataKey="avg_ttft" name="平均首字 (s)" stroke="#38bdf8" strokeWidth={1.75} dot={false} />
                    </ComposedChart>
                  </ResponsiveContainer>
                )}
              </div>
            )}

            {/* 底部图例 */}
            {!dualChartCompare && (
              <div className="mt-3 flex items-center gap-5 text-xs text-muted-foreground flex-wrap">
                {chartMode === 'token' && (
                  <>
                    <span className="inline-flex items-center gap-1.5">
                      <span className="w-2.5 h-2.5 rounded-sm bg-[#ef7d52]" />
                      Token 总量
                    </span>
                    <span className="inline-flex items-center gap-1.5">
                      <span className="w-2.5 h-2.5 rounded-full bg-[#7c9cbf]" />
                      请求数
                    </span>
                  </>
                )}
                {chartMode === 'cache' && (
                  <>
                    <span className="inline-flex items-center gap-1.5">
                      <span className="w-2.5 h-2.5 rounded-sm bg-sky-400" />
                      输入（命中缓存）
                    </span>
                    <span className="inline-flex items-center gap-1.5">
                      <span className="w-2.5 h-2.5 rounded-sm bg-amber-500" />
                      输入（未命中缓存）
                    </span>
                    <span className="inline-flex items-center gap-1.5">
                      <span className="w-2.5 h-2.5 rounded-sm bg-emerald-500" />
                      输出 Token
                    </span>
                    <span className="inline-flex items-center gap-1.5">
                      <span className="w-2.5 h-2.5 rounded-full bg-[#e07a5f]" />
                      缓存命中率 (%)
                    </span>
                  </>
                )}
                {chartMode === 'requests' && (
                  <>
                    <span className="inline-flex items-center gap-1.5">
                      <span className="w-2.5 h-2.5 rounded-sm bg-emerald-500" />
                      成功请求
                    </span>
                    <span className="inline-flex items-center gap-1.5">
                      <span className="w-2.5 h-2.5 rounded-sm bg-red-500" />
                      失败请求
                    </span>
                    <span className="inline-flex items-center gap-1.5">
                      <span className="w-2.5 h-2.5 rounded-full bg-sky-400" />
                      成功率 (%)
                    </span>
                  </>
                )}
                {chartMode === 'throughput' && (
                  <>
                    <span className="inline-flex items-center gap-1.5">
                      <span className="w-2.5 h-2.5 rounded-sm bg-amber-500" />
                      生成速度 (tok/s)
                    </span>
                    <span className="inline-flex items-center gap-1.5">
                      <span className="w-2.5 h-2.5 rounded-full bg-sky-400" />
                      平均首字耗时 (s)
                    </span>
                  </>
                )}
              </div>
            )}
          </div>

          {/* 右栏：最近请求 实时活动流（点击任意请求的模型或渠道可直接筛选） */}
          <div
            className={`${
              dualChartCompare ? 'xl:col-span-12' : 'xl:col-span-5'
            } bg-card border border-border/80 rounded-2xl p-3.5 sm:p-5 shadow-sm flex flex-col justify-between min-w-0 overflow-hidden`}
          >
            <div className="flex flex-wrap items-center justify-between gap-1 mb-3">
              <div className="flex items-baseline gap-2 flex-wrap">
                <h3 className="text-sm font-bold tracking-wide text-foreground">最近请求</h3>
                <span className="text-[11px] sm:text-xs text-muted-foreground">点击模型/渠道/Key 可筛选</span>
              </div>
              <span className="text-xs text-muted-foreground tabular-nums">
                共 {recentTotal.toLocaleString()} 条记录
              </span>
            </div>

            <div className="overflow-y-auto max-h-[300px] pr-1 divide-y divide-border/50">
              {recentRequests.length === 0 ? (
                <div className="py-16 text-center text-xs text-muted-foreground">
                  暂无最近请求记录
                </div>
              ) : (
                recentRequests.map(item => (
                  <div
                    key={item.id}
                    className="py-2 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-1 sm:gap-2 text-xs hover:bg-muted/30 px-1.5 rounded-md transition-colors min-w-0"
                  >
                    <div className="flex items-center justify-between sm:justify-start gap-2 min-w-0">
                      <div className="flex items-center gap-1.5 min-w-0">
                        <span className="font-mono text-[11px] text-muted-foreground tabular-nums shrink-0">
                          {formatClockTime(item.timestamp)}
                        </span>
                        <span
                          className={`w-1.5 h-1.5 rounded-full shrink-0 ${
                            item.success ? 'bg-emerald-500' : 'bg-rose-500'
                          }`}
                          title={`HTTP ${item.status_code}`}
                        />
                        {item.key_label && (
                          <button
                            type="button"
                            onClick={() => item.key_hash && toggleKeyFilter(item.key_hash)}
                            className={`sm:hidden inline-flex items-center gap-0.5 px-1.5 py-0.2 rounded text-[10px] border transition-colors shrink-0 ${
                              item.key_hash && selectedKeys.includes(item.key_hash)
                                ? 'bg-amber-500/20 text-amber-400 border-amber-500/40 font-semibold'
                                : 'bg-muted/50 text-muted-foreground border-border/60 hover:text-foreground'
                            }`}
                            title={`API Key: ${item.key_label}${item.api_key_prefix ? ` (${item.api_key_prefix})` : ''} · 点击只看该 Key`}
                          >
                            <KeyRound className="w-2.5 h-2.5 text-amber-400" />
                            <span className="max-w-[90px] truncate">{item.key_label}</span>
                          </button>
                        )}
                      </div>
                      <div className="sm:hidden shrink-0 font-mono text-[11px] tabular-nums text-right">
                        <span className="font-semibold text-foreground">
                          {formatTokens(item.total_tokens)}
                        </span>{' '}
                        <span className="text-muted-foreground">tok</span>
                        <span className="text-muted-foreground"> · {item.process_time.toFixed(1)}s</span>
                      </div>
                    </div>

                    <div className="flex items-center justify-between gap-2 min-w-0 flex-1">
                      <div className="flex items-center gap-1 min-w-0 flex-wrap sm:flex-nowrap">
                        <button
                          type="button"
                          onClick={() => toggleModelFilter(item.model)}
                          className="font-semibold text-foreground hover:text-primary transition-colors truncate max-w-[170px] sm:max-w-[160px]"
                          title={`点击只看模型 ${item.model}`}
                        >
                          {item.model}
                        </button>
                        <span className="text-muted-foreground shrink-0">→</span>
                        <button
                          type="button"
                          onClick={() => toggleProviderFilter(item.provider)}
                          className="text-muted-foreground hover:text-foreground transition-colors truncate max-w-[130px] sm:max-w-[120px]"
                          title={`点击只看渠道 ${item.provider}`}
                        >
                          {item.provider}
                        </button>
                        {item.key_label && (
                          <button
                            type="button"
                            onClick={() => item.key_hash && toggleKeyFilter(item.key_hash)}
                            className={`hidden sm:inline-flex ml-1 items-center gap-0.5 px-1.5 py-0.2 rounded text-[10px] border transition-colors shrink-0 ${
                              item.key_hash && selectedKeys.includes(item.key_hash)
                                ? 'bg-amber-500/20 text-amber-400 border-amber-500/40 font-semibold'
                                : 'bg-muted/50 text-muted-foreground border-border/60 hover:text-foreground'
                            }`}
                            title={`API Key: ${item.key_label}${item.api_key_prefix ? ` (${item.api_key_prefix})` : ''} · 点击只看该 Key`}
                          >
                            <KeyRound className="w-2.5 h-2.5 text-amber-400" />
                            <span className="max-w-[85px] truncate">{item.key_label}</span>
                          </button>
                        )}
                      </div>

                      <div className="hidden sm:block shrink-0 font-mono text-[11px] tabular-nums text-right">
                        <span className="font-semibold text-foreground">
                          {formatTokens(item.total_tokens)}
                        </span>{' '}
                        <span className="text-muted-foreground">tok</span>
                        <span className="text-muted-foreground"> · {item.process_time.toFixed(1)}s</span>
                      </div>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </div>

      {/* =====================================================================
          2.5 API Key 用量排行与明细看板（查看哪条 Key 用量最多、Token/费用占比、缓存命中与一键下钻）
         ===================================================================== */}
      <div
        id="section-key-usage"
        className="bg-card border border-border/80 rounded-2xl p-3.5 sm:p-5 shadow-sm space-y-4 min-w-0 overflow-hidden"
      >
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2 flex-wrap">
            <KeyRound className="w-4 h-4 text-amber-400 shrink-0" />
            <h3 className="text-sm font-bold text-foreground">API Key 用量排行</h3>
            <span className="text-xs text-muted-foreground">
              （共 {keyBreakdown.length} 条活跃 Key · 点击任意 Key 联动筛选，点击「深度分析」查看 IP 与配额）
            </span>
          </div>
          {selectedKeys.length > 0 && (
            <button
              type="button"
              onClick={() => setSelectedKeys([])}
              className="inline-flex items-center gap-1 px-2.5 py-1 text-xs font-medium text-amber-400 bg-amber-500/10 hover:bg-amber-500/20 border border-amber-500/30 rounded-lg transition-colors"
            >
              <X className="w-3 h-3" />
              取消 Key 过滤 ({selectedKeys.length})
            </button>
          )}
        </div>

        {keyBreakdown.length === 0 ? (
          <div className="py-10 text-center text-xs text-muted-foreground border border-dashed border-border/60 rounded-xl">
            当前时间范围与筛选条件下暂无 API Key 用量记录
          </div>
        ) : (
          <div className="grid grid-cols-1 xl:grid-cols-12 gap-4 items-start min-w-0">
            {/* 左栏：Top Key Token & 费用占比可视化卡片 */}
            <div className="xl:col-span-5 bg-muted/20 border border-border/60 rounded-xl p-3 sm:p-4 space-y-3 min-w-0">
              <div className="flex flex-wrap items-center justify-between gap-1 text-xs">
                <span className="font-semibold text-foreground">Top Key 消耗占比</span>
                <span className="text-muted-foreground tabular-nums">
                  总计 {formatTokens(summary?.total_tokens || 0)} tok · ¥{(summary?.total_cost_cny || 0).toFixed(2)}
                </span>
              </div>
              <div className="space-y-2.5">
                {keyBreakdown.slice(0, 6).map((k, idx) => {
                  const isSelected = selectedKeys.includes(k.key_hash);
                  return (
                    <div
                      key={k.key_hash}
                      onClick={() => toggleKeyFilter(k.key_hash)}
                      className={`p-2.5 rounded-xl border transition-all cursor-pointer min-w-0 ${
                        isSelected
                          ? 'bg-amber-500/10 border-amber-500/50 shadow-sm'
                          : 'bg-card/80 border-border/60 hover:border-border'
                      }`}
                    >
                      <div className="flex items-start justify-between gap-2 text-xs min-w-0">
                        <div className="flex items-center gap-1.5 flex-wrap min-w-0 flex-1">
                          <span
                            className="w-5 h-5 rounded-md flex items-center justify-center text-[10px] font-bold text-white shrink-0"
                            style={{ backgroundColor: LINE_COLORS[idx % LINE_COLORS.length] }}
                          >
                            {idx + 1}
                          </span>
                          <span className="font-semibold text-foreground truncate max-w-[140px] sm:max-w-[180px]" title={k.key_label}>
                            {k.key_label}
                          </span>
                          {k.api_key_name && (
                            <span className="text-[10px] font-mono text-muted-foreground shrink-0">
                              ({k.api_key_prefix})
                            </span>
                          )}
                          {k.api_key_group && (
                            <span className="px-1.5 py-0.2 rounded bg-muted text-muted-foreground text-[10px] shrink-0">
                              {k.api_key_group}
                            </span>
                          )}
                          {k.parent_name && (
                            <span
                              className="px-1.5 py-0.2 rounded bg-purple-500/10 text-purple-400 text-[10px] shrink-0"
                              title={`子 Key，继承自母 Key: ${k.parent_name}`}
                            >
                              母:{k.parent_name}
                            </span>
                          )}
                        </div>
                        <div className="shrink-0 font-mono text-xs tabular-nums flex items-center gap-1.5">
                          <span>
                            <span className="font-bold text-foreground">{formatTokens(k.total_tokens)}</span>
                            <span className="text-muted-foreground ml-0.5">({k.token_share.toFixed(1)}%)</span>
                          </span>
                          {k.can_drilldown && (
                            <button
                              type="button"
                              onClick={e => {
                                e.stopPropagation();
                                setAnalyticsTargetKey({
                                  key_hash: k.key_hash,
                                  key_label: k.key_label,
                                });
                              }}
                              className="xl:hidden px-1.5 py-0.5 text-[10px] font-sans font-medium rounded bg-primary/10 text-primary border border-primary/30 whitespace-nowrap"
                            >
                              分析
                            </button>
                          )}
                        </div>
                      </div>

                      <div className="mt-2 h-1.5 w-full bg-muted rounded-full overflow-hidden">
                        <div
                          className="h-full rounded-full transition-all duration-500"
                          style={{
                            width: `${Math.max(2, Math.min(100, k.token_share))}%`,
                            backgroundColor: LINE_COLORS[idx % LINE_COLORS.length],
                          }}
                        />
                      </div>

                      <div className="mt-1.5 flex flex-wrap items-center justify-between gap-x-2 gap-y-1 text-[11px] text-muted-foreground tabular-nums">
                        <span className="min-w-0 break-all sm:break-normal">
                          请求 <strong className="text-foreground/90">{k.requests.toLocaleString()}</strong> 次 · 缓存{' '}
                          <strong className="text-cyan-400">{k.cache_hit_rate.toFixed(1)}%</strong> · 主力{' '}
                          <strong className="text-foreground/90">{k.top_model}</strong>
                        </span>
                        <span className="font-mono text-rose-400 font-semibold shrink-0">
                          ¥{k.cost_cny.toFixed(2)} (${k.cost_usd.toFixed(2)})
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* 右栏：Key 用量速览与下钻表 */}
            <div className="xl:col-span-7 overflow-x-auto border border-border/70 rounded-xl min-w-0 w-full">
              <table className="w-full min-w-[680px] text-left text-xs whitespace-nowrap">
                <thead className="bg-muted/60 text-muted-foreground font-medium">
                  <tr>
                    <th className="px-3.5 py-2.5">API Key 标识</th>
                    <th className="px-3.5 py-2.5 text-right">请求数</th>
                    <th className="px-3.5 py-2.5 text-right">成功率</th>
                    <th className="px-3.5 py-2.5 text-right">输入 / 输出</th>
                    <th className="px-3.5 py-2.5 text-right">缓存命中率</th>
                    <th className="px-3.5 py-2.5 text-right">总 Token</th>
                    <th className="px-3.5 py-2.5 text-right">等价费用</th>
                    <th className="px-3.5 py-2.5 text-right">操作</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border/50 tabular-nums">
                  {keyBreakdown.slice(0, 10).map(row => {
                    const isSelected = selectedKeys.includes(row.key_hash);
                    return (
                      <tr
                        key={row.key_hash}
                        onClick={() => toggleKeyFilter(row.key_hash)}
                        className={`cursor-pointer transition-colors ${
                          isSelected ? 'bg-amber-500/10' : 'hover:bg-muted/30'
                        }`}
                        title="点击按该 Key 联动筛选全页"
                      >
                        <td className="px-3.5 py-2.5">
                          <div className="flex items-center gap-1.5">
                            <KeyRound className="w-3.5 h-3.5 text-amber-400 shrink-0" />
                            <div>
                              <div className="font-semibold text-foreground flex items-center gap-1.5">
                                <span>{row.key_label}</span>
                                {row.api_key_group && (
                                  <span className="px-1.5 py-0.2 rounded bg-muted text-muted-foreground text-[10px] font-normal">
                                    {row.api_key_group}
                                  </span>
                                )}
                              </div>
                              <div className="text-[10px] text-muted-foreground font-mono">
                                {row.api_key_prefix}
                                {row.top_model && row.top_model !== '-' ? ` · ${row.top_model}` : ''}
                              </div>
                            </div>
                          </div>
                        </td>
                        <td className="px-3.5 py-2.5 text-right text-foreground font-medium">
                          {row.requests.toLocaleString()}
                        </td>
                        <td className="px-3.5 py-2.5 text-right">
                          <span
                            className={`font-semibold ${
                              row.success_rate >= 95
                                ? 'text-emerald-400'
                                : row.success_rate >= 80
                                ? 'text-amber-400'
                                : 'text-rose-400'
                            }`}
                          >
                            {row.success_rate.toFixed(1)}%
                          </span>
                        </td>
                        <td className="px-3.5 py-2.5 text-right text-muted-foreground">
                          {formatTokens(row.prompt_tokens)} / {formatTokens(row.completion_tokens)}
                        </td>
                        <td className="px-3.5 py-2.5 text-right">
                          <span className="text-cyan-400 font-semibold">{row.cache_hit_rate.toFixed(1)}%</span>
                        </td>
                        <td className="px-3.5 py-2.5 text-right font-mono font-bold text-foreground">
                          {formatTokens(row.total_tokens)}
                        </td>
                        <td className="px-3.5 py-2.5 text-right font-mono text-rose-400 font-semibold">
                          ¥{row.cost_cny.toFixed(2)}
                        </td>
                        <td
                          className="px-3.5 py-2.5 text-right"
                          onClick={e => e.stopPropagation()}
                        >
                          {row.can_drilldown ? (
                            <button
                              type="button"
                              onClick={() =>
                                setAnalyticsTargetKey({
                                  key_hash: row.key_hash,
                                  key_label: row.key_label,
                                })
                              }
                              className="px-2 py-1 text-[11px] font-medium rounded-md bg-primary/10 text-primary hover:bg-primary/20 border border-primary/30 transition-colors whitespace-nowrap"
                            >
                              深度分析
                            </button>
                          ) : (
                            <span className="text-[11px] text-muted-foreground">—</span>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>

      {/* =====================================================================
          3. 渠道与模型分布三联图（原版经典 3 图同屏：渠道成功率 + 模型分布 + Endpoint 分布）
         ===================================================================== */}
      <div id="section-distributions" className="grid grid-cols-1 lg:grid-cols-12 gap-4 min-w-0">
        {/* 渠道成功率 (%) */}
        <div className="lg:col-span-5 bg-card border border-border/80 rounded-2xl p-3.5 sm:p-5 shadow-sm min-w-0 overflow-hidden">
          <div className="flex items-center justify-between gap-2 mb-4">
            <h3 className="text-sm font-bold text-foreground flex items-center gap-2">
              <Cpu className="w-4 h-4 text-emerald-500 shrink-0" />
              渠道成功率 (%)
            </h3>
            <span className="text-[11px] text-muted-foreground">点击柱子可筛选渠道</span>
          </div>
          <div className="h-56 sm:h-60 min-w-0">
            {formattedChannelStats.length === 0 ? (
              <div className="h-full flex items-center justify-center text-xs text-muted-foreground">
                暂无渠道数据
              </div>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={formattedChannelStats} margin={{ top: 0, right: 0, left: -12, bottom: 0 }}>
                  <XAxis dataKey="name" stroke={AXIS_COLOR} fontSize={11} tickLine={false} axisLine={false} />
                  <YAxis stroke={AXIS_COLOR} fontSize={11} tickLine={false} axisLine={false} domain={[0, 100]} />
                  <Tooltip
                    cursor={{ fill: 'hsl(var(--muted) / 0.5)' }}
                    contentStyle={tooltipStyle}
                    formatter={(value: number) => [`${value.toFixed(1)}%`, '成功率']}
                  />
                  <Bar
                    dataKey="success_rate"
                    name="成功率"
                    radius={[4, 4, 0, 0]}
                    className="cursor-pointer"
                    onClick={(data: any) => data?.name && toggleProviderFilter(data.name)}
                  >
                    {formattedChannelStats.map((entry, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={
                          entry.success_rate >= 95
                            ? SUCCESS_COLOR
                            : entry.success_rate >= 80
                            ? WARNING_COLOR
                            : ERROR_COLOR
                        }
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>

        {/* 模型请求量分布 */}
        <div className="lg:col-span-4 bg-card border border-border/80 rounded-2xl p-3.5 sm:p-5 shadow-sm min-w-0 overflow-hidden">
          <div className="flex items-center justify-between gap-2 mb-4">
            <h3 className="text-sm font-bold text-foreground flex items-center gap-2">
              <Activity className="w-4 h-4 text-blue-500 shrink-0" />
              模型请求量分布
            </h3>
            <span className="text-[11px] text-muted-foreground">点击扇区可筛选模型</span>
          </div>
          <div className="h-60 min-w-0">
            {modelPieData.length === 0 ? (
              <div className="h-full flex items-center justify-center text-xs text-muted-foreground">
                暂无模型请求数据
              </div>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={modelPieData}
                    cx="50%"
                    cy="44%"
                    innerRadius={40}
                    outerRadius={68}
                    paddingAngle={3}
                    dataKey="count"
                    nameKey="model"
                    className="cursor-pointer"
                    onClick={(data: any) => data?.model && toggleModelFilter(data.model)}
                  >
                    {modelPieData.map((_, index) => (
                      <Cell key={`cell-${index}`} fill={CHART_COLORS[index % CHART_COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip contentStyle={tooltipStyle} />
                  <Legend
                    verticalAlign="bottom"
                    align="center"
                    wrapperStyle={{ fontSize: '10px', paddingTop: '4px' }}
                    formatter={(value: string) => (
                      <span className="text-foreground truncate inline-block max-w-[115px] align-bottom" title={value}>
                        {value}
                      </span>
                    )}
                  />
                </PieChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>

        {/* 接口访问分布 (Endpoint) */}
        <div className="lg:col-span-3 bg-card border border-border/80 rounded-2xl p-3.5 sm:p-5 shadow-sm min-w-0 overflow-hidden">
          <h3 className="text-sm font-bold text-foreground mb-4 flex items-center gap-2">
            <Server className="w-4 h-4 text-purple-500 shrink-0" />
            接口访问分布 (Endpoint)
          </h3>
          <div className="h-60 min-w-0">
            {formattedEndpointStats.length === 0 ? (
              <div className="h-full flex items-center justify-center text-xs text-muted-foreground">
                暂无接口数据
              </div>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={formattedEndpointStats}
                    cx="50%"
                    cy="44%"
                    innerRadius={36}
                    outerRadius={66}
                    paddingAngle={2}
                    dataKey="value"
                    nameKey="name"
                    label={({ percent }) => `${(percent * 100).toFixed(0)}%`}
                    labelLine={false}
                  >
                    {formattedEndpointStats.map((_, index) => (
                      <Cell key={`cell-${index}`} fill={CHART_COLORS[(index + 2) % CHART_COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip contentStyle={tooltipStyle} />
                  <Legend
                    verticalAlign="bottom"
                    wrapperStyle={{ fontSize: '10px' }}
                    formatter={(value: string) => (
                      <span className="text-muted-foreground truncate inline-block max-w-[110px] align-bottom" title={value}>
                        {value}
                      </span>
                    )}
                  />
                </PieChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>
      </div>

      {/* =====================================================================
          4. 渠道健康状况详细表（原版表格 + 融合缓存率、速度、Token、成本）
         ===================================================================== */}
      <div
        id="section-channel-health"
        className="bg-card border border-border/80 rounded-2xl shadow-sm overflow-hidden min-w-0"
      >
        <div className="px-3.5 sm:px-5 py-3.5 sm:py-4 border-b border-border/70 bg-muted/20 flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2 flex-wrap">
            <Cpu className="w-4 h-4 text-primary shrink-0" />
            <h3 className="text-sm font-bold text-foreground">渠道健康状况详细</h3>
            <span className="text-xs text-muted-foreground">
              （基于 channel_stats 真实尝试成功率 + 遥测性能指标）
            </span>
          </div>
          <span className="text-xs text-muted-foreground tabular-nums">
            共 {channelStats.length} 个渠道
          </span>
        </div>
        <div className="overflow-x-auto max-h-[360px] w-full">
          <table className="w-full min-w-[680px] text-left text-xs whitespace-nowrap">
            <thead className="bg-muted/70 text-muted-foreground font-medium sticky top-0 z-10">
              <tr>
                <th className="px-4 sm:px-5 py-3">渠道名称</th>
                <th className="px-4 sm:px-5 py-3">健康状态</th>
                <th className="px-4 sm:px-5 py-3 text-right">尝试请求数</th>
                <th className="px-4 sm:px-5 py-3 text-right">渠道成功率</th>
                <th className="px-4 sm:px-5 py-3 text-right">缓存命中率</th>
                <th className="px-4 sm:px-5 py-3 text-right">生成速度 / 首字</th>
                <th className="px-4 sm:px-5 py-3 text-right">Token 用量</th>
                <th className="px-4 sm:px-5 py-3 text-right">API 等价</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border/50 tabular-nums">
              {channelStats.length === 0 ? (
                <tr>
                  <td colSpan={8} className="px-5 py-8 text-center text-muted-foreground">
                    暂无渠道数据
                  </td>
                </tr>
              ) : (
                channelStats.map((channel, i) => {
                  const isHealthy = channel.success_rate >= 0.95;
                  const isWarning = channel.success_rate < 0.95 && channel.success_rate >= 0.8;
                  const extra = providerExtraMap.get(channel.provider);
                  const isSelected = selectedProviders.includes(channel.provider);
                  return (
                    <tr
                      key={i}
                      onClick={() => toggleProviderFilter(channel.provider)}
                      className={`cursor-pointer transition-colors ${
                        isSelected ? 'bg-primary/10' : 'hover:bg-muted/40'
                      }`}
                      title="点击筛选该渠道"
                    >
                      <td className="px-4 sm:px-5 py-3 font-semibold text-foreground">{channel.provider}</td>
                      <td className="px-4 sm:px-5 py-3">
                        <span
                          className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-medium border ${
                            isHealthy
                              ? 'bg-emerald-500/10 text-emerald-500 border-emerald-500/20'
                              : isWarning
                              ? 'bg-amber-500/10 text-amber-500 border-amber-500/20'
                              : 'bg-red-500/10 text-red-500 border-red-500/20'
                          }`}
                        >
                          {isHealthy ? (
                            <CheckCircle2 className="w-3 h-3" />
                          ) : (
                            <AlertCircle className="w-3 h-3" />
                          )}
                          {isHealthy ? '良好' : isWarning ? '警告' : '异常'}
                        </span>
                      </td>
                      <td className="px-4 sm:px-5 py-3 text-right text-muted-foreground">
                        {channel.total_requests.toLocaleString()}
                      </td>
                      <td className="px-4 sm:px-5 py-3 text-right font-mono font-bold">
                        <span
                          className={
                            isHealthy ? 'text-emerald-500' : isWarning ? 'text-amber-500' : 'text-red-500'
                          }
                        >
                          {(channel.success_rate * 100).toFixed(1)}%
                        </span>
                      </td>
                      <td className="px-4 sm:px-5 py-3 text-right text-cyan-400 font-medium">
                        {extra ? `${extra.cache_hit_rate.toFixed(1)}%` : '-'}
                      </td>
                      <td className="px-4 sm:px-5 py-3 text-right text-muted-foreground">
                        {extra && extra.speed_tok_s > 0
                          ? `${extra.speed_tok_s.toFixed(1)} tok/s · ${extra.avg_ttft.toFixed(1)}s`
                          : '-'}
                      </td>
                      <td className="px-4 sm:px-5 py-3 text-right font-semibold text-foreground">
                        {extra ? formatTokens(extra.total_tokens) : '-'}
                      </td>
                      <td className="px-4 sm:px-5 py-3 text-right text-rose-400 font-semibold">
                        {extra ? `¥${extra.cost_cny.toFixed(2)}` : '-'}
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* =====================================================================
          5. 工作回溯 · 近一年每日活动热力图（受顶部全局模型/渠道筛选联动）
         ===================================================================== */}
      <div
        id="section-heatmap"
        className="bg-card border border-border/80 rounded-2xl p-3.5 sm:p-5 shadow-sm space-y-4 min-w-0 overflow-hidden"
      >
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-baseline gap-2.5 flex-wrap">
            <h3 className="text-sm font-bold tracking-wide text-foreground">工作回溯</h3>
            <span className="text-xs text-muted-foreground">
              近一年每日活动 {hasActiveFilter ? '（已按顶部所选模型/渠道过滤）' : ''}
            </span>
          </div>
          <div className="text-xs text-muted-foreground tabular-nums">
            {hoveredDay ? (
              <span className="text-foreground font-medium">
                {hoveredDay.date} · {hoveredDay.requests.toLocaleString()} 次请求 · {formatTokens(hoveredDay.total_tokens)} tokens
              </span>
            ) : (
              <span>
                {heatmap?.active_days || 0} 天有活动 · 共 {(heatmap?.total_requests || 0).toLocaleString()} 次请求
              </span>
            )}
          </div>
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-12 gap-4 sm:gap-6 items-center min-w-0">
          <div className="xl:col-span-7 overflow-x-auto pb-1 min-w-0 w-full">
            <div className="inline-block min-w-[640px]">
              <div className="relative h-4 mb-1 ml-6 text-[10px] text-muted-foreground select-none">
                {heatmapGrid.monthHeaders.map((mh, idx) => (
                  <span key={idx} className="absolute" style={{ left: `${mh.colIndex * 13}px` }}>
                    {mh.label}
                  </span>
                ))}
              </div>

              <div className="flex items-start gap-1.5">
                <div className="grid grid-rows-7 gap-[3px] text-[10px] text-muted-foreground pr-1 select-none leading-[10px]">
                  <span className="h-[10px] flex items-center">一</span>
                  <span className="h-[10px]" />
                  <span className="h-[10px] flex items-center">三</span>
                  <span className="h-[10px]" />
                  <span className="h-[10px] flex items-center">五</span>
                  <span className="h-[10px]" />
                  <span className="h-[10px]" />
                </div>

                <div className="flex gap-[3px]">
                  {heatmapGrid.weeks.map((week, wIdx) => (
                    <div key={wIdx} className="grid grid-rows-7 gap-[3px]">
                      {week.map((day, dIdx) =>
                        day === null ? (
                          <span key={dIdx} className="w-[10px] h-[10px] rounded-[2px] opacity-0" />
                        ) : (
                          <span
                            key={day.date}
                            onMouseEnter={() => setHoveredDay(day)}
                            onMouseLeave={() => setHoveredDay(null)}
                            title={`${day.date}: ${day.requests} 请求 / ${formatTokens(day.total_tokens)} tokens`}
                            className={`w-[10px] h-[10px] rounded-[2px] transition-transform hover:scale-125 cursor-pointer ${getHeatmapCellColor(
                              day.requests,
                              heatmapGrid.maxReqs
                            )}`}
                          />
                        )
                      )}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          <div className="xl:col-span-5 grid grid-cols-2 gap-y-3 sm:gap-y-5 gap-x-4 sm:gap-x-8 pt-2 xl:pt-0 xl:pl-4 border-t xl:border-t-0 xl:border-l border-border/60 min-w-0">
            <div>
              <div className="text-lg sm:text-xl font-bold tabular-nums text-foreground">
                {heatmap?.active_days || 0}
              </div>
              <div className="text-xs text-muted-foreground mt-0.5">天有活动</div>
            </div>
            <div>
              <div className="text-lg sm:text-xl font-bold tabular-nums text-foreground">
                {(heatmap?.total_requests || 0).toLocaleString()}
              </div>
              <div className="text-xs text-muted-foreground mt-0.5">总请求</div>
            </div>
            <div>
              <div className="text-lg sm:text-xl font-bold tabular-nums text-foreground">
                {formatTokens(heatmap?.total_tokens || 0)}
              </div>
              <div className="text-xs text-muted-foreground mt-0.5">总 Token</div>
            </div>
            <div>
              <div className="text-lg sm:text-xl font-bold tabular-nums text-foreground">
                {(heatmap?.peak_day_requests || 0).toLocaleString()}
              </div>
              <div className="text-xs text-muted-foreground mt-0.5">
                峰值单日 · {heatmap?.peak_day_date || '--'}
              </div>
            </div>
          </div>
        </div>

        <div className="flex items-center justify-between text-[11px] text-muted-foreground pt-1">
          <span>近一年 {formatTokens(heatmap?.total_tokens || 0)} tokens</span>
          <div className="flex items-center gap-1">
            <span>少</span>
            <span className="w-2.5 h-2.5 rounded-[2px] bg-muted/70 dark:bg-[#2a2927]" />
            <span className="w-2.5 h-2.5 rounded-[2px] bg-[#6b4c3e]" />
            <span className="w-2.5 h-2.5 rounded-[2px] bg-[#9a5b42]" />
            <span className="w-2.5 h-2.5 rounded-[2px] bg-[#c96a48]" />
            <span className="w-2.5 h-2.5 rounded-[2px] bg-[#ef7d52]" />
            <span>多</span>
          </div>
        </div>
      </div>

      {/* =====================================================================
          6. 统一用量明细表（渠道×模型交叉明细 / 按模型汇总 / 按渠道汇总）
             - 直接联动顶部筛选器，去除重复的底部查询表单
         ===================================================================== */}
      <div
        id="section-breakdown"
        className="bg-card border border-border/80 rounded-2xl p-3.5 sm:p-5 shadow-sm space-y-4 min-w-0 overflow-hidden"
      >
        <div className="flex flex-col sm:flex-row sm:flex-wrap sm:items-center sm:justify-between gap-2.5 sm:gap-3 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <BarChart3 className="w-4 h-4 text-blue-500 shrink-0" />
            <h3 className="text-sm font-bold text-foreground">用量明细</h3>
            <span className="text-xs text-muted-foreground">
              （点击表格任意模型或渠道可一键联动全页筛选）
            </span>
          </div>

          <div className="flex flex-col sm:flex-row sm:items-center gap-2 sm:flex-wrap min-w-0 w-full sm:w-auto">
            {/* 表内快速搜索 */}
            <div className="relative w-full sm:w-48">
              <Search className="w-3.5 h-3.5 text-muted-foreground absolute left-2.5 top-1/2 -translate-y-1/2" />
              <input
                type="text"
                value={tableSearch}
                onChange={e => setTableSearch(e.target.value)}
                placeholder="搜索 Key、模型或渠道..."
                className="pl-8 pr-6 py-1.5 sm:py-1 text-xs bg-background border border-border rounded-lg text-foreground w-full focus:outline-none focus:ring-1 focus:ring-primary"
              />
              {tableSearch && (
                <button
                  type="button"
                  onClick={() => setTableSearch('')}
                  className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                >
                  <X className="w-3 h-3" />
                </button>
              )}
            </div>

            <div className="inline-flex items-center bg-muted/50 border border-border/80 rounded-lg p-0.5 gap-0.5 overflow-x-auto no-scrollbar max-w-full flex-nowrap sm:flex-wrap min-w-0">
              {[
                { key: 'cross', label: `渠道 × 模型 (${filteredCrossRows.length})` },
                { key: 'key', label: `按 Key 汇总 (${filteredKeyRows.length})` },
                { key: 'key_model', label: `Key × 模型 (${filteredKeyModelRows.length})` },
                { key: 'model', label: `按模型汇总 (${filteredModelRows.length})` },
                { key: 'provider', label: `按渠道汇总 (${filteredProviderRows.length})` },
              ].map(tab => (
                <button
                  key={tab.key}
                  type="button"
                  onClick={() =>
                    setBreakdownTab(tab.key as 'cross' | 'key' | 'key_model' | 'model' | 'provider')
                  }
                  className={`px-2.5 sm:px-3 py-1 text-xs font-medium rounded-md transition-all whitespace-nowrap shrink-0 ${
                    breakdownTab === tab.key
                      ? 'bg-card text-foreground shadow-sm border border-border/60 font-semibold'
                      : 'text-muted-foreground hover:text-foreground'
                  }`}
                >
                  {tab.label}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* 渠道 × 模型交叉明细表 */}
        {breakdownTab === 'cross' && (
          <div className="overflow-x-auto border border-border/70 rounded-xl w-full">
            <table className="w-full min-w-[720px] text-left text-xs whitespace-nowrap">
              <thead className="bg-muted/60 text-muted-foreground font-medium">
                <tr>
                  <th className="px-4 py-3">渠道</th>
                  <th className="px-4 py-3">模型</th>
                  <th className="px-4 py-3 text-right">请求次数</th>
                  <th className="px-4 py-3 text-right">成功率</th>
                  <th className="px-4 py-3 text-right">输入 Token</th>
                  <th className="px-4 py-3 text-right">输出 Token</th>
                  <th className="px-4 py-3 text-right">缓存命中率</th>
                  <th className="px-4 py-3 text-right">生成速度</th>
                  <th className="px-4 py-3 text-right">总 Token</th>
                  <th className="px-4 py-3 text-right">实记费用 / 等价</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/50 tabular-nums">
                {filteredCrossRows.length === 0 ? (
                  <tr>
                    <td colSpan={10} className="px-4 py-8 text-center text-muted-foreground">
                      暂无匹配的明细记录
                    </td>
                  </tr>
                ) : (
                  filteredCrossRows.map((entry, i) => (
                    <tr key={i} className="hover:bg-muted/30 transition-colors">
                      <td className="px-4 py-3 font-medium text-foreground">
                        <button
                          type="button"
                          onClick={() => toggleProviderFilter(entry.provider)}
                          className="hover:text-primary transition-colors"
                        >
                          {entry.provider}
                        </button>
                      </td>
                      <td className="px-4 py-3 font-mono">
                        <button
                          type="button"
                          onClick={() => toggleModelFilter(entry.model)}
                          className="text-foreground font-semibold hover:text-primary transition-colors"
                        >
                          {entry.model}
                        </button>
                      </td>
                      <td className="px-4 py-3 text-right text-muted-foreground">
                        {entry.requests.toLocaleString()}
                      </td>
                      <td className="px-4 py-3 text-right">
                        <span
                          className={`font-semibold ${
                            entry.success_rate >= 95
                              ? 'text-emerald-400'
                              : entry.success_rate >= 80
                              ? 'text-amber-400'
                              : 'text-rose-400'
                          }`}
                        >
                          {entry.success_rate.toFixed(1)}%
                        </span>
                      </td>
                      <td className="px-4 py-3 text-right text-muted-foreground">
                        {formatTokens(entry.prompt_tokens)}
                      </td>
                      <td className="px-4 py-3 text-right text-muted-foreground">
                        {formatTokens(entry.completion_tokens)}
                      </td>
                      <td className="px-4 py-3 text-right">
                        <span className="text-cyan-400 font-semibold">{entry.cache_hit_rate.toFixed(1)}%</span>
                        <span className="text-muted-foreground ml-1">({formatTokens(entry.cached_tokens)})</span>
                      </td>
                      <td className="px-4 py-3 text-right text-muted-foreground">
                        {entry.speed_tok_s > 0 ? `${entry.speed_tok_s.toFixed(1)} tok/s` : '-'}
                      </td>
                      <td className="px-4 py-3 text-right font-mono font-bold text-foreground">
                        {formatTokens(entry.total_tokens)}
                      </td>
                      <td className="px-4 py-3 text-right font-mono">
                        <span className="text-amber-500 font-semibold">
                          {formatCost(entry.recorded_cost_usd)}
                        </span>
                        <span className="text-rose-400 ml-1.5">(¥{entry.cost_cny.toFixed(2)})</span>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        )}

        {/* 按 Key 汇总明细表 */}
        {breakdownTab === 'key' && (
          <div className="overflow-x-auto border border-border/70 rounded-xl w-full">
            <table className="w-full min-w-[780px] text-left text-xs whitespace-nowrap">
              <thead className="bg-muted/60 text-muted-foreground font-medium">
                <tr>
                  <th className="px-4 py-3">API Key 名称 / 前缀</th>
                  <th className="px-4 py-3">分组 / 母 Key</th>
                  <th className="px-4 py-3 text-right">请求数</th>
                  <th className="px-4 py-3 text-right">成功率</th>
                  <th className="px-4 py-3 text-right">输入 Token</th>
                  <th className="px-4 py-3 text-right">输出 Token</th>
                  <th className="px-4 py-3 text-right">缓存命中率</th>
                  <th className="px-4 py-3 text-right">速度 / 首字</th>
                  <th className="px-4 py-3 text-right">总 Token (占比)</th>
                  <th className="px-4 py-3 text-right">实记费用 / 等价</th>
                  <th className="px-4 py-3 text-right">详情</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/50 tabular-nums">
                {filteredKeyRows.length === 0 ? (
                  <tr>
                    <td colSpan={11} className="px-4 py-8 text-center text-muted-foreground">
                      暂无匹配的 API Key 用量记录
                    </td>
                  </tr>
                ) : (
                  filteredKeyRows.map(row => (
                    <tr
                      key={row.key_hash}
                      onClick={() => toggleKeyFilter(row.key_hash)}
                      className={`hover:bg-muted/30 cursor-pointer transition-colors ${
                        selectedKeys.includes(row.key_hash) ? 'bg-amber-500/10' : ''
                      }`}
                    >
                      <td className="px-4 py-3">
                        <div className="font-semibold text-foreground">{row.key_label}</div>
                        <div className="text-[10px] font-mono text-muted-foreground">
                          {row.api_key_prefix} · {row.unique_models} 模型
                        </div>
                      </td>
                      <td className="px-4 py-3 text-muted-foreground">
                        {row.api_key_group || '-'}
                        {row.parent_name ? ` (母: ${row.parent_name})` : ''}
                      </td>
                      <td className="px-4 py-3 text-right">{row.requests.toLocaleString()}</td>
                      <td className="px-4 py-3 text-right font-semibold text-emerald-400">
                        {row.success_rate.toFixed(1)}%
                      </td>
                      <td className="px-4 py-3 text-right text-muted-foreground">
                        {formatTokens(row.prompt_tokens)}
                      </td>
                      <td className="px-4 py-3 text-right text-muted-foreground">
                        {formatTokens(row.completion_tokens)}
                      </td>
                      <td className="px-4 py-3 text-right">
                        <span className="text-cyan-400 font-bold">{row.cache_hit_rate.toFixed(1)}%</span>
                        <span className="text-muted-foreground ml-1">({formatTokens(row.cached_tokens)})</span>
                      </td>
                      <td className="px-4 py-3 text-right text-muted-foreground">
                        <span className="text-foreground font-medium">{row.speed_tok_s.toFixed(1)}</span> tok/s ·{' '}
                        {row.avg_ttft.toFixed(1)}s
                      </td>
                      <td className="px-4 py-3 text-right">
                        <div className="flex items-center justify-end gap-2">
                          <span className="font-semibold text-foreground">{formatTokens(row.total_tokens)}</span>
                          <div className="w-14 h-1.5 bg-muted rounded-full overflow-hidden">
                            <div
                              className="h-full bg-amber-500"
                              style={{ width: `${Math.min(100, row.token_share)}%` }}
                            />
                          </div>
                        </div>
                      </td>
                      <td className="px-4 py-3 text-right font-mono">
                        <span className="text-amber-500 font-semibold">
                          {formatCost(row.recorded_cost_usd)}
                        </span>
                        <span className="text-rose-400 ml-1.5">(¥{row.cost_cny.toFixed(2)})</span>
                      </td>
                      <td
                        className="px-4 py-3 text-right"
                        onClick={e => e.stopPropagation()}
                      >
                        {row.can_drilldown ? (
                          <button
                            type="button"
                            onClick={() =>
                              setAnalyticsTargetKey({
                                key_hash: row.key_hash,
                                key_label: row.key_label,
                              })
                            }
                            className="px-2 py-0.5 text-[11px] font-medium rounded bg-primary/10 text-primary hover:bg-primary/20 border border-primary/30"
                          >
                            分析
                          </button>
                        ) : (
                          '-'
                        )}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        )}

        {/* Key × 模型交叉明细表 */}
        {breakdownTab === 'key_model' && (
          <div className="overflow-x-auto border border-border/70 rounded-xl w-full">
            <table className="w-full min-w-[720px] text-left text-xs whitespace-nowrap">
              <thead className="bg-muted/60 text-muted-foreground font-medium">
                <tr>
                  <th className="px-4 py-3">API Key</th>
                  <th className="px-4 py-3">模型</th>
                  <th className="px-4 py-3 text-right">请求次数</th>
                  <th className="px-4 py-3 text-right">成功率</th>
                  <th className="px-4 py-3 text-right">输入 Token</th>
                  <th className="px-4 py-3 text-right">输出 Token</th>
                  <th className="px-4 py-3 text-right">缓存命中率</th>
                  <th className="px-4 py-3 text-right">生成速度</th>
                  <th className="px-4 py-3 text-right">总 Token</th>
                  <th className="px-4 py-3 text-right">实记费用 / 等价</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/50 tabular-nums">
                {filteredKeyModelRows.length === 0 ? (
                  <tr>
                    <td colSpan={10} className="px-4 py-8 text-center text-muted-foreground">
                      暂无匹配的 Key × 模型明细记录
                    </td>
                  </tr>
                ) : (
                  filteredKeyModelRows.map((entry, i) => (
                    <tr key={i} className="hover:bg-muted/30 transition-colors">
                      <td className="px-4 py-3 font-medium text-foreground">
                        <button
                          type="button"
                          onClick={() => toggleKeyFilter(entry.key_hash)}
                          className="hover:text-amber-400 transition-colors text-left"
                        >
                          <div className="font-semibold">{entry.key_label}</div>
                          <div className="text-[10px] font-mono text-muted-foreground">
                            {entry.api_key_prefix}
                          </div>
                        </button>
                      </td>
                      <td className="px-4 py-3 font-mono">
                        <button
                          type="button"
                          onClick={() => toggleModelFilter(entry.model)}
                          className="text-foreground font-semibold hover:text-primary transition-colors"
                        >
                          {entry.model}
                        </button>
                      </td>
                      <td className="px-4 py-3 text-right text-muted-foreground">
                        {entry.requests.toLocaleString()}
                      </td>
                      <td className="px-4 py-3 text-right">
                        <span
                          className={`font-semibold ${
                            entry.success_rate >= 95
                              ? 'text-emerald-400'
                              : entry.success_rate >= 80
                              ? 'text-amber-400'
                              : 'text-rose-400'
                          }`}
                        >
                          {entry.success_rate.toFixed(1)}%
                        </span>
                      </td>
                      <td className="px-4 py-3 text-right text-muted-foreground">
                        {formatTokens(entry.prompt_tokens)}
                      </td>
                      <td className="px-4 py-3 text-right text-muted-foreground">
                        {formatTokens(entry.completion_tokens)}
                      </td>
                      <td className="px-4 py-3 text-right">
                        <span className="text-cyan-400 font-semibold">{entry.cache_hit_rate.toFixed(1)}%</span>
                        <span className="text-muted-foreground ml-1">({formatTokens(entry.cached_tokens)})</span>
                      </td>
                      <td className="px-4 py-3 text-right text-muted-foreground">
                        {entry.speed_tok_s > 0 ? `${entry.speed_tok_s.toFixed(1)} tok/s` : '-'}
                      </td>
                      <td className="px-4 py-3 text-right font-mono font-bold text-foreground">
                        {formatTokens(entry.total_tokens)}
                      </td>
                      <td className="px-4 py-3 text-right font-mono">
                        <span className="text-amber-500 font-semibold">
                          {formatCost(entry.recorded_cost_usd)}
                        </span>
                        <span className="text-rose-400 ml-1.5">(¥{entry.cost_cny.toFixed(2)})</span>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        )}

        {/* 按模型汇总明细表 */}
        {breakdownTab === 'model' && (
          <div className="overflow-x-auto border border-border/70 rounded-xl w-full">
            <table className="w-full min-w-[680px] text-left text-xs whitespace-nowrap">
              <thead className="bg-muted/60 text-muted-foreground font-medium">
                <tr>
                  <th className="px-4 py-3">模型名称</th>
                  <th className="px-4 py-3 text-right">请求数</th>
                  <th className="px-4 py-3 text-right">成功率</th>
                  <th className="px-4 py-3 text-right">输入 Token</th>
                  <th className="px-4 py-3 text-right">输出 Token</th>
                  <th className="px-4 py-3 text-right">独立缓存命中率</th>
                  <th className="px-4 py-3 text-right">速度 / 首字</th>
                  <th className="px-4 py-3 text-right">总 Token (占比)</th>
                  <th className="px-4 py-3 text-right">API 等价</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/50 tabular-nums">
                {filteredModelRows.map(row => (
                  <tr
                    key={row.model}
                    onClick={() => row.model && toggleModelFilter(row.model)}
                    className="hover:bg-muted/30 cursor-pointer transition-colors"
                  >
                    <td className="px-4 py-3 font-semibold text-foreground">{row.model}</td>
                    <td className="px-4 py-3 text-right">{row.requests.toLocaleString()}</td>
                    <td className="px-4 py-3 text-right font-semibold text-emerald-400">
                      {row.success_rate.toFixed(1)}%
                    </td>
                    <td className="px-4 py-3 text-right text-muted-foreground">
                      {formatTokens(row.prompt_tokens)}
                    </td>
                    <td className="px-4 py-3 text-right text-muted-foreground">
                      {formatTokens(row.completion_tokens)}
                    </td>
                    <td className="px-4 py-3 text-right">
                      <span className="text-cyan-400 font-bold">{row.cache_hit_rate.toFixed(1)}%</span>
                      <span className="text-muted-foreground ml-1">({formatTokens(row.cached_tokens)})</span>
                    </td>
                    <td className="px-4 py-3 text-right text-muted-foreground">
                      <span className="text-foreground font-medium">{row.speed_tok_s.toFixed(1)}</span> tok/s ·{' '}
                      {row.avg_ttft.toFixed(1)}s
                    </td>
                    <td className="px-4 py-3 text-right">
                      <div className="flex items-center justify-end gap-2">
                        <span className="font-semibold text-foreground">{formatTokens(row.total_tokens)}</span>
                        <div className="w-14 h-1.5 bg-muted rounded-full overflow-hidden">
                          <div
                            className="h-full bg-[#ef7d52]"
                            style={{ width: `${Math.min(100, row.token_share)}%` }}
                          />
                        </div>
                      </div>
                    </td>
                    <td className="px-4 py-3 text-right font-semibold text-rose-400">
                      ¥{row.cost_cny.toFixed(2)} (${row.cost_usd.toFixed(2)})
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* 按渠道汇总明细表 */}
        {breakdownTab === 'provider' && (
          <div className="overflow-x-auto border border-border/70 rounded-xl w-full">
            <table className="w-full min-w-[640px] text-left text-xs whitespace-nowrap">
              <thead className="bg-muted/60 text-muted-foreground font-medium">
                <tr>
                  <th className="px-4 py-3">渠道名称</th>
                  <th className="px-4 py-3 text-right">请求数</th>
                  <th className="px-4 py-3 text-right">成功率</th>
                  <th className="px-4 py-3 text-right">缓存命中率</th>
                  <th className="px-4 py-3 text-right">平均速度 / 首字</th>
                  <th className="px-4 py-3 text-right">总 Token</th>
                  <th className="px-4 py-3 text-right">API 等价</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/50 tabular-nums">
                {filteredProviderRows.map(row => (
                  <tr
                    key={row.provider}
                    onClick={() => row.provider && toggleProviderFilter(row.provider)}
                    className="hover:bg-muted/30 cursor-pointer transition-colors"
                  >
                    <td className="px-4 py-3 font-semibold text-foreground">{row.provider}</td>
                    <td className="px-4 py-3 text-right">{row.requests.toLocaleString()}</td>
                    <td className="px-4 py-3 text-right font-semibold text-emerald-400">
                      {row.success_rate.toFixed(1)}%
                    </td>
                    <td className="px-4 py-3 text-right text-cyan-400 font-bold">
                      {row.cache_hit_rate.toFixed(1)}%
                    </td>
                    <td className="px-4 py-3 text-right text-muted-foreground">
                      <span className="text-foreground font-medium">{row.speed_tok_s.toFixed(1)}</span> tok/s ·{' '}
                      {row.avg_ttft.toFixed(1)}s
                    </td>
                    <td className="px-4 py-3 text-right font-semibold text-foreground">
                      {formatTokens(row.total_tokens)}
                    </td>
                    <td className="px-4 py-3 text-right font-semibold text-rose-400">
                      ¥{row.cost_cny.toFixed(2)} (${row.cost_usd.toFixed(2)})
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Key 深度分析侧滑抽屉（查看 IP 分布、配额与报错明细，全程使用安全 key_hash 不暴露完整密钥） */}
      <KeyAnalyticsSheet
        open={Boolean(analyticsTargetKey)}
        onOpenChange={open => {
          if (!open) setAnalyticsTargetKey(null);
        }}
        apiKeyHash={analyticsTargetKey?.key_hash}
        apiKeyName={analyticsTargetKey?.key_label}
      />
    </div>
  );
}
