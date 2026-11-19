import { useCallback, useEffect, useMemo, useState } from 'react';

const API_BASE = __API_BASE_URL__.replace(/\/$/, '');
const SEGMENT_COLORS = ['#4d69e8', '#ef9c45', '#24a58a', '#ae78d5'];
const FEATURES = [
  { key: 'Total_Orders', label: 'Total orders', type: 'number', min: 1, max: 1000, step: 1 },
  { key: 'Average_Order_Value', label: 'Average order value', type: 'number', min: 0.01, max: 1000000, step: 0.01 },
  { key: 'Days_Since_Last_Order', label: 'Days since last order', type: 'number', min: 0, max: 10000, step: 1 },
  { key: 'Average_Days_Between_Orders', label: 'Average days between orders', type: 'number', min: 0, max: 10000, step: 1 },
  { key: 'Preferred_Category_Share', label: 'Preferred category share', type: 'number', min: 0.01, max: 1, step: 0.01 },
];

async function api(path, params = {}, init = {}) {
  const query = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value !== '' && value !== null && value !== undefined) query.set(key, String(value));
  });
  const url = `${API_BASE}${path}${query.size ? `?${query}` : ''}`;
  let response;
  try {
    response = await fetch(url, {
      ...init,
      headers: { ...(init.body ? { 'Content-Type': 'application/json' } : {}), ...init.headers },
    });
  } catch {
    throw new Error(`Could not reach the API at ${API_BASE}. Start the backend and check the API base URL.`);
  }
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = body.details?.map((item) => item.message).join(' ');
    throw new Error(detail || body.message || `API request failed (${response.status}).`);
  }
  return body;
}

const money = (value, compact = false) => {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—';
  return new Intl.NumberFormat('en-US', {
    style: 'currency', currency: 'USD', maximumFractionDigits: compact ? 1 : 0,
    notation: compact ? 'compact' : 'standard',
  }).format(value);
};
const number = (value) => new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 }).format(value ?? 0);
const percent = (value) => `${Number(value || 0).toFixed(1)}%`;

function Icon({ name }) {
  const paths = {
    grid: <><rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/></>,
    spark: <><path d="m12 3 1.9 5.8L20 11l-6.1 2.2L12 19l-2-5.8L4 11l6-2.2L12 3Z"/><path d="m19 15 .9 2.1L22 18l-2.1.9L19 21l-.9-2.1L16 18l2.1-.9L19 15Z"/></>,
    chart: <><path d="M4 19V5M4 19h17"/><path d="m7 15 4-4 3 2 6-7"/></>,
    users: <><path d="M16 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="10" cy="7" r="4"/><path d="M20 21v-2a4 4 0 0 0-3-3.9M16 3.1a4 4 0 0 1 0 7.8"/></>,
    arrow: <><path d="M7 17 17 7M7 7h10v10"/></>,
    filter: <><path d="M4 7h16M7 12h10m-7 5h4"/><circle cx="8" cy="7" r="2"/><circle cx="15" cy="12" r="2"/></>,
  };
  return <svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">{paths[name] || paths.grid}</svg>;
}

function Loading({ label = 'Loading dashboard' }) {
  return <div className="loading"><span className="spinner" />{label}</div>;
}

function Empty({ children = 'No data for this selection.' }) {
  return <div className="empty-state">{children}</div>;
}

function TrendChart({ points }) {
  const [mode, setMode] = useState('revenue');
  const width = 720;
  const height = 220;
  const pad = { top: 14, right: 12, bottom: 34, left: 12 };
  const values = points.map((point) => mode === 'revenue' ? point.revenue : point.orders);
  const max = Math.max(...values, 1);
  const coords = values.map((value, index) => ({
    x: pad.left + (points.length < 2 ? (width - pad.left - pad.right) / 2 : index * (width - pad.left - pad.right) / (points.length - 1)),
    y: pad.top + (height - pad.top - pad.bottom) * (1 - value / max),
  }));
  const line = coords.map((point, index) => `${index ? 'L' : 'M'} ${point.x} ${point.y}`).join(' ');
  const area = coords.length ? `${line} L ${coords.at(-1).x} ${height - pad.bottom} L ${coords[0].x} ${height - pad.bottom} Z` : '';
  const step = Math.max(1, Math.ceil(points.length / 6));
  if (!points.length) return <Empty />;
  return <>
    <div className="chart-switch" role="group" aria-label="Chart metric">
      {['revenue', 'orders'].map((item) => <button key={item} className={mode === item ? 'selected' : ''} onClick={() => setMode(item)}>{item}</button>)}
    </div>
    <div className="trend-chart">
      <div className="chart-y-labels"><span>{mode === 'revenue' ? money(max, true) : number(max)}</span><span>{mode === 'revenue' ? money(max / 2, true) : number(max / 2)}</span><span>0</span></div>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${mode} trend`} preserveAspectRatio="none">
        <defs><linearGradient id="trendFill" x1="0" x2="0" y1="0" y2="1"><stop offset="0%" stopColor="#536fe8" stopOpacity=".2"/><stop offset="100%" stopColor="#536fe8" stopOpacity="0"/></linearGradient></defs>
        {[0, 0.5, 1].map((fraction) => <line key={fraction} x1={pad.left} x2={width - pad.right} y1={pad.top + (height - pad.top - pad.bottom) * fraction} y2={pad.top + (height - pad.top - pad.bottom) * fraction} className="gridline"/>)}
        <path d={area} fill="url(#trendFill)"/><path d={line} className="trend-line"/>
        {coords.map((point, index) => <circle key={points[index].period} cx={point.x} cy={point.y} r="3.5" className="trend-dot"><title>{points[index].period}: {mode === 'revenue' ? money(points[index].revenue) : number(points[index].orders)}</title></circle>)}
      </svg>
      <div className="chart-x-labels">{points.map((point, index) => index % step === 0 || index === points.length - 1 ? <span key={point.period}>{point.period}</span> : <span key={point.period} />)}</div>
    </div>
  </>;
}

function KpiCard({ icon, label, value, note, tone }) {
  return <article className="kpi-card"><div className={`kpi-icon ${tone}`}><Icon name={icon}/></div><div className="kpi-label">{label}</div><div className="kpi-value">{value}</div><div className="kpi-note">{note}</div></article>;
}

function BarList({ items, nameKey, valueKey, formatter = money }) {
  const max = Math.max(...items.map((item) => item[valueKey] || 0), 1);
  if (!items.length) return <Empty />;
  return <div className="bar-list">{items.slice(0, 7).map((item, index) => <div className="bar-row" key={item[nameKey]}>
    <div className="bar-meta"><span title={item[nameKey]}>{item[nameKey]}</span><strong>{formatter(item[valueKey])}</strong></div>
    <div className="bar-track"><span style={{ width: `${Math.max(2, (item[valueKey] / max) * 100)}%`, animationDelay: `${index * 45}ms` }} /></div>
  </div>)}</div>;
}

function Dashboard({ data, onSegment, onReload, loading, filters, draft, setDraft, options, applyFilters, clearFilters, error }) {
  const { summary, trend, categories, customers, segments } = data || {};
  const kpi = summary?.kpis || {};
  const filterInputs = [
    { key: 'start_date', label: 'From', type: 'date' },
    { key: 'end_date', label: 'To', type: 'date' },
  ];
  return <main className="main-content">
    <div className="page-heading">
      <div><div className="eyebrow">STORE PERFORMANCE</div><h1>Good morning <span className="wave">✦</span></h1><p>Here’s what’s happening across your store.</p></div>
      <button className="button button-soft" onClick={onReload}><span className="refresh-symbol">↻</span> Refresh data</button>
    </div>
    <section className="filter-panel">
      <div className="filter-title"><span className="filter-icon"><Icon name="filter"/></span><div><strong>Filters</strong><small>Refine your dashboard view</small></div></div>
      {filterInputs.map((field) => <label className="filter-field" key={field.key}><span>{field.label}</span><input type={field.type} value={draft[field.key]} min={field.key === 'start_date' ? options?.date_min : undefined} max={field.key === 'end_date' ? options?.date_max : undefined} onChange={(event) => setDraft((current) => ({ ...current, [field.key]: event.target.value }))}/></label>)}
      <label className="filter-field"><span>Category</span><select value={draft.category} onChange={(event) => setDraft((current) => ({ ...current, category: event.target.value }))}><option value="">All categories</option>{(options?.categories || []).map((value) => <option key={value}>{value}</option>)}</select></label>
      <label className="filter-field"><span>City</span><select value={draft.city} onChange={(event) => setDraft((current) => ({ ...current, city: event.target.value }))}><option value="">All cities</option>{(options?.cities || []).map((value) => <option key={value}>{value}</option>)}</select></label>
      <button className="button button-primary filter-apply" onClick={applyFilters}>Apply filters</button>
      <button className="clear-button" onClick={clearFilters}>Clear</button>
    </section>
    {error && <div className="alert alert-error"><strong>Dashboard couldn’t load.</strong> {error}<button onClick={onReload}>Retry</button></div>}
    {loading && !data ? <Loading /> : <>
      <div className="section-heading"><div><h2>At a glance</h2><p>{summary?.date_range ? `${summary.date_range.start} — ${summary.date_range.end}` : 'Selected period'}</p></div><span className="live-pill"><i/> Live data</span></div>
      <section className="kpi-grid">
        <KpiCard icon="chart" label="Total revenue" value={money(kpi.total_revenue, true)} note="Across all orders" tone="blue"/>
        <KpiCard icon="grid" label="Total orders" value={number(kpi.total_orders)} note="Orders in selected period" tone="orange"/>
        <KpiCard icon="users" label="Customers" value={number(kpi.total_customers)} note="Unique customers" tone="green"/>
        <KpiCard icon="spark" label="Average order value" value={money(kpi.average_order_value)} note={`${Number(kpi.average_customer_rating || 0).toFixed(1)} / 5 average rating`} tone="purple"/>
      </section>
      <section className="analytics-grid">
        <article className="panel trend-panel"><div className="panel-heading"><div><h3>Revenue over time</h3><p>Monthly sales performance</p></div><span className="panel-badge">15 months</span></div><TrendChart points={trend?.points || []}/></article>
        <article className="panel category-panel"><div className="panel-heading"><div><h3>Category performance</h3><p>Revenue by product category</p></div><button className="icon-button" aria-label="Open categories">···</button></div><BarList items={categories?.categories || []} nameKey="category" valueKey="revenue" formatter={(value) => money(value, true)}/></article>
      </section>
      <section className="lower-grid">
        <article className="panel customer-panel"><div className="panel-heading"><div><h3>Customer health</h3><p>One-time vs returning buyers</p></div><span className="panel-badge">{number(customers?.total_customers)} customers</span></div>
          <div className="customer-summary"><div className="donut" style={{ '--repeat': `${customers?.repeat_pct || 0}%` }}><div><strong>{percent(customers?.repeat_pct)}</strong><span>returning</span></div></div><div className="customer-legend"><div><i className="legend-dot new-dot"/><span>Single-order</span><strong>{percent(customers?.single_order_pct)}</strong></div><div><i className="legend-dot repeat-dot"/><span>Repeat buyers</span><strong>{percent(customers?.repeat_pct)}</strong></div><small>Average rating: {Number(kpi.average_customer_rating || 0).toFixed(2)} / 5</small></div></div>
        </article>
        <article className="panel segments-panel"><div className="panel-heading"><div><h3>Customer segments</h3><p>Production model · 4 groups</p></div><button className="text-link" onClick={() => document.getElementById('segment-grid')?.scrollIntoView({ behavior: 'smooth' })}>View all <Icon name="arrow"/></button></div>
          <div className="mini-segments">{(segments?.segments || []).map((segment) => <button className="mini-segment" key={segment.cluster} onClick={() => onSegment(segment.cluster)}><span className="segment-swatch" style={{ background: SEGMENT_COLORS[segment.cluster] }}/><span className="mini-segment-copy"><strong>{segment.segment}</strong><small>{percent(segment.customers_pct)} of customers</small></span><b>{number(segment.customers)}</b></button>)}</div>
        </article>
      </section>
      <section className="panel full-panel" id="segment-grid"><div className="panel-heading"><div><h3>Segment overview</h3><p>Customer value and buying behavior from saved production segmentation</p></div><span className="model-chip"><Icon name="spark"/> K-means · K=4</span></div>
        <div className="segment-grid">{(segments?.segments || []).map((segment) => <article className="segment-card" key={segment.cluster} style={{ '--segment-color': SEGMENT_COLORS[segment.cluster] }}><div className="segment-card-top"><span className="segment-number">0{segment.cluster + 1}</span><span className="segment-share">{percent(segment.customers_pct)}</span></div><h4>{segment.segment}</h4><p>{segment.description}</p><div className="segment-stats"><div><small>Customers</small><strong>{number(segment.customers)}</strong></div><div><small>Revenue share</small><strong>{percent(segment.revenue_share_pct)}</strong></div></div><button className="segment-open" onClick={() => onSegment(segment.cluster)}>Explore segment <span>→</span></button></article>)}</div>
      </section>
      <div className="footnote">Average quantity per order: {Number(kpi.average_quantity_per_order || 0).toFixed(2)} · API: {API_BASE}</div>
    </>}
  </main>;
}

function Prediction({ result, setResult }) {
  const [form, setForm] = useState({ Total_Orders: '1', Average_Order_Value: '1150', Days_Since_Last_Order: '45', Average_Days_Between_Orders: '', Preferred_Category_Share: '1' });
  const [pending, setPending] = useState(false);
  const [error, setError] = useState('');
  const update = (key, value) => setForm((current) => ({ ...current, [key]: value }));
  const submit = async (event) => {
    event.preventDefault(); setPending(true); setError(''); setResult(null);
    const orders = Number(form.Total_Orders);
    const payload = {
      Total_Orders: orders,
      Average_Order_Value: Number(form.Average_Order_Value),
      Days_Since_Last_Order: Number(form.Days_Since_Last_Order),
      Average_Days_Between_Orders: orders === 1 ? null : Number(form.Average_Days_Between_Orders),
      Preferred_Category_Share: Number(form.Preferred_Category_Share),
    };
    try { setResult(await api('/api/segmentation/predict', {}, { method: 'POST', body: JSON.stringify(payload) })); }
    catch (issue) { setError(issue.message); }
    finally { setPending(false); }
  };
  return <main className="main-content prediction-page">
    <div className="page-heading"><div><div className="eyebrow">CUSTOMER INTELLIGENCE</div><h1>Segment prediction</h1><p>See where a customer fits across your four production segments.</p></div><div className="model-status"><i/> Model ready</div></div>
    <div className="prediction-layout"><section className="panel prediction-form-panel"><div className="panel-heading"><div><h3>Customer profile</h3><p>Enter the five features used by the saved model.</p></div><span className="model-chip"><Icon name="spark"/> Core 5 model</span></div>
      <form onSubmit={submit} className="prediction-form">
        {FEATURES.map((feature) => <label className="input-field" key={feature.key}><span>{feature.label}<small>{feature.key === 'Preferred_Category_Share' ? '0–1 share' : feature.key === 'Average_Order_Value' ? 'Currency per order' : ' '}</small></span><input required type={feature.type} min={feature.min} max={feature.max} step={feature.step} value={form[feature.key]} disabled={feature.key === 'Average_Days_Between_Orders' && Number(form.Total_Orders) === 1} placeholder={feature.key === 'Average_Days_Between_Orders' && Number(form.Total_Orders) === 1 ? 'Not needed for one order' : ''} onChange={(event) => update(feature.key, event.target.value)}/></label>)}
        <div className="form-hint">For a single-order customer, average days between orders is set to null automatically.</div>
        {error && <div className="alert alert-error">{error}</div>}
        <button className="button button-primary predict-button" disabled={pending}>{pending ? <><span className="spinner small-spinner"/> Analyzing customer</> : <><Icon name="spark"/> Predict customer segment</>}</button>
      </form>
    </section>
    <section className="panel result-panel">{result ? <PredictionResult result={result}/> : <div className="result-placeholder"><span className="placeholder-orbit"><Icon name="spark"/></span><h3>Your result will appear here</h3><p>Run a prediction to see the segment match, assignment strength, and suggested next steps.</p><div className="placeholder-facts"><span><b>4</b> customer groups</span><span><b>5</b> model features</span><span><b>0</b> retraining steps</span></div></div>}</section></div>
    <div className="prediction-note"><Icon name="spark"/><span><strong>Production model</strong> Predictions use the saved Core 5 K-means model and preprocessor from <code>models/</code>. The model is never retrained by this screen.</span></div>
  </main>;
}

function PredictionResult({ result }) {
  const color = SEGMENT_COLORS[result.cluster] || SEGMENT_COLORS[0];
  return <div className="prediction-result" style={{ '--segment-color': color }}>
    <div className="result-kicker">PREDICTED SEGMENT</div><div className="result-title-row"><span className="result-segment-icon"><Icon name="users"/></span><div><h3>{result.segment}</h3><p>Cluster {result.cluster} · {result.is_single_order_customer ? 'Single-order customer' : 'Repeat customer'}</p></div></div>
    <p className="result-description">{result.description}</p>
    <div className="assignment-meter"><div><span>Assignment confidence</span><strong>{result.assignment.strength}</strong></div><div className="meter-track"><span style={{ width: `${Math.min(100, Math.max(0, result.assignment.margin * 100))}%` }}/></div><small>Nearest alternative: {result.assignment.closest_alternative.segment}</small></div>
    <div className="result-profile"><div><small>Typical orders</small><strong>{result.segment_profile.avg_orders.toFixed(1)}</strong></div><div><small>Avg order value</small><strong>{money(result.segment_profile.avg_order_value)}</strong></div><div><small>Revenue share</small><strong>{percent(result.segment_profile.revenue_share_pct)}</strong></div></div>
    <div className="action-list"><h4>Recommended next steps</h4>{result.suggested_actions.map((action, index) => <div className="action-item" key={action}><span>{String(index + 1).padStart(2, '0')}</span>{action}</div>)}</div>
    {result.warnings?.length > 0 && <div className="warning-box"><strong>Review these notes</strong>{result.warnings.map((warning) => <p key={warning}>{warning}</p>)}</div>}
  </div>;
}

function SegmentModal({ detail, onClose, loading, error }) {
  useEffect(() => {
    const handler = (event) => { if (event.key === 'Escape') onClose(); };
    window.addEventListener('keydown', handler); return () => window.removeEventListener('keydown', handler);
  }, [onClose]);
  return <div className="modal-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}><section className="segment-modal" role="dialog" aria-modal="true" aria-label="Segment details">
    <button className="modal-close" aria-label="Close" onClick={onClose}>×</button>{loading ? <Loading label="Loading segment"/> : error ? <div className="alert alert-error">{error}</div> : detail && <>
      <div className="eyebrow">SEGMENT 0{detail.segment.cluster + 1}</div><h2>{detail.segment.segment}</h2><p className="modal-description">{detail.segment.description}</p>
      <div className="modal-stats"><div><small>Customers</small><strong>{number(detail.segment.customers)}</strong></div><div><small>Revenue</small><strong>{money(detail.segment.revenue, true)}</strong></div><div><small>Avg orders</small><strong>{Number(detail.segment.average_orders || 0).toFixed(1)}</strong></div></div>
      <h3>Suggested actions</h3><ul className="suggestion-list">{detail.suggested_actions.map((action) => <li key={action}>{action}</li>)}</ul>
      <h3>Top categories</h3><BarList items={detail.category_mix} nameKey="name" valueKey="revenue" formatter={(value) => money(value, true)}/>
    </>}</section></div>;
}

export default function App() {
  const [page, setPage] = useState('dashboard');
  const [data, setData] = useState(null);
  const [options, setOptions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [filters, setFilters] = useState({ start_date: '', end_date: '', category: '', city: '' });
  const [draft, setDraft] = useState(filters);
  const [detail, setDetail] = useState(null);
  const [detailState, setDetailState] = useState({ loading: false, error: '' });
  const [prediction, setPrediction] = useState(null);

  const loadDashboard = useCallback(async (activeFilters = filters) => {
    setLoading(true); setError('');
    try {
      const [summary, trend, categories, customers, segments] = await Promise.all([
        api('/api/dashboard/summary', activeFilters),
        api('/api/dashboard/sales-trend', { ...activeFilters, granularity: 'month' }),
        api('/api/dashboard/category-performance', activeFilters),
        api('/api/dashboard/customer-analysis', activeFilters),
        api('/api/dashboard/segments', activeFilters),
      ]);
      setData({ summary, trend, categories, customers, segments });
    } catch (issue) { setError(issue.message); }
    finally { setLoading(false); }
  }, [filters]);

  useEffect(() => {
    api('/api/dashboard/filters').then(setOptions).catch((issue) => setError(issue.message));
    loadDashboard();
  }, []);

  const applyFilters = () => {
    if (draft.start_date && draft.end_date && draft.start_date > draft.end_date) {
      setError('Start date must be on or before end date.'); return;
    }
    setFilters(draft); loadDashboard(draft);
  };
  const clearFilters = () => {
    const clean = { start_date: '', end_date: '', category: '', city: '' };
    setDraft(clean); setFilters(clean); loadDashboard(clean);
  };
  const openSegment = async (id) => {
    setDetail(null); setDetailState({ loading: true, error: '' });
    try { const value = await api(`/api/dashboard/segment/${id}`, filters); setDetail(value); }
    catch (issue) { setDetailState({ loading: false, error: issue.message }); return; }
    setDetailState({ loading: false, error: '' });
  };
  const closeSegment = useCallback(() => setDetail(null), []);
  const navItems = useMemo(() => [
    { id: 'dashboard', label: 'Overview', icon: 'grid' },
    { id: 'predict', label: 'Segment prediction', icon: 'spark' },
  ], []);

  return <div className="app-shell">
    <aside className="sidebar"><a className="brand" href="#overview" onClick={(event) => { event.preventDefault(); setPage('dashboard'); }}><span className="brand-mark"><span/><span/><span/><span/></span><span>commerce<span className="brand-light">intel</span><small>ANALYTICS PLATFORM</small></span></a>
      <div className="nav-label">WORKSPACE</div><nav>{navItems.map((item) => <button key={item.id} className={`nav-item ${page === item.id ? 'active' : ''}`} onClick={() => setPage(item.id)}><Icon name={item.icon}/><span>{item.label}</span>{item.id === 'dashboard' && <i className="nav-indicator"/>}</button>)}</nav>
      <div className="sidebar-bottom"><div className="sidebar-status"><span className="status-dot"/><div><strong>API connection</strong><small>{error ? 'Needs attention' : 'Connected to local API'}</small></div></div><div className="sidebar-footer">Stage 8 <span>·</span> Customer intelligence</div></div>
    </aside>
    <div className="content-wrap"><header className="topbar"><div className="breadcrumb"><span>Workspace</span><b>/</b><strong>{page === 'dashboard' ? 'Overview' : 'Segment prediction'}</strong></div><div className="topbar-right"><span className="topbar-date">E-commerce analytics</span><span className="avatar">CI</span></div></header>
      {page === 'dashboard' ? <Dashboard data={data} loading={loading} error={error} filters={filters} draft={draft} setDraft={setDraft} options={options} applyFilters={applyFilters} clearFilters={clearFilters} onReload={() => loadDashboard(filters)} onSegment={openSegment}/> : <Prediction result={prediction} setResult={setPrediction}/>}
    </div>
    {(detail || detailState.loading || detailState.error) && <SegmentModal detail={detail} loading={detailState.loading} error={detailState.error} onClose={closeSegment}/>}
  </div>;
}
