/* dashboard.js — BaseballBetBot GitHub Pages dashboard */
'use strict';

// ── State ──────────────────────────────────────────────────────────────────
let allHistory  = [];
let statsData   = {};
let chart       = null;
let mode        = 'ytd';        // 'ytd' | 'last30' | 'all_time'
let market      = 'all';        // 'moneyline' | 'totals' | 'all'
let filterText  = '';
let todayPicks  = [];           // today's moneyline slate (cached for sort/filter)
let todayTotals = [];           // today's totals slate
const slateState = {            // per-slate sort + tier filter (presentational only)
  picks:  { sort: 'edge', tier: 'all' },
  totals: { sort: 'edge', tier: 'all' },
};

// ── Fetch ──────────────────────────────────────────────────────────────────
async function loadData() {
  try {
    const [statsRes, historyRes, picksRes, totalsRes] = await Promise.all([
      fetch('data/stats.json'),
      fetch('data/picks_history.json'),
      fetch('data/picks_today.json'),
      fetch('data/totals_today.json'),
    ]);

    if (!statsRes.ok || !historyRes.ok) {
      throw new Error('Could not load dashboard data.');
    }

    statsData  = await statsRes.json();
    allHistory = await historyRes.json();   // all markets; filtered per-view
    const rawToday = picksRes.ok ? await picksRes.json() : [];
    const rawTotals = totalsRes && totalsRes.ok ? await totalsRes.json() : [];
    // Only show picks that are actually for today (US Eastern). Between runs the
    // committed picks_today.json still holds the previous day's picks; without
    // this guard the site would display yesterday's matchups as "Today's Picks".
    const today = easternDateStr();
    todayPicks  = rawToday.filter(p => p.date === today);
    todayTotals = rawTotals.filter(p => p.date === today);

    const loadingEl = document.getElementById('loading');
    const appEl     = document.getElementById('app');
    if (loadingEl) loadingEl.style.display = 'none';
    if (appEl)     appEl.style.display     = 'block';

    renderLastUpdated(statsData.last_updated);
    renderModelStatus(statsData.model);
    renderStats();
    renderChart(marketFiltered(allHistory), mode);
    renderTable(allHistory);
    renderPickOfDay(todayPicks);
    renderTodayPicks();
    renderTodayTotals();

  } catch (err) {
    const loadingEl = document.getElementById('loading');
    if (loadingEl) loadingEl.style.display = 'none';
    showError('Could not load data: ' + err.message);
    console.error(err);
  }
}

// ── Helpers ────────────────────────────────────────────────────────────────
function $(id) { return document.getElementById(id); }

// Current date (YYYY-MM-DD) in US Eastern — matches how picks are dated and
// keeps a day's picks visible through that day in the league's reference zone.
function easternDateStr() {
  return new Date().toLocaleDateString('en-CA', { timeZone: 'America/New_York' });
}

// Apply the active market filter (moneyline / totals / all) to history rows.
function marketFiltered(history) {
  if (market === 'all') return history;
  if (market === 'totals') return history.filter(p => p.bet_type === 'totals');
  return history.filter(p => !p.bet_type || p.bet_type === 'moneyline');
}

// True when a row falls inside the active time range.
function inTimeRange(p, targetMode) {
  if (!p.date) return false;
  if (targetMode === 'ytd') {
    return p.date.startsWith(String(new Date().getFullYear()));
  }
  if (targetMode === 'last30') {
    const cutoff = new Date(Date.now() - 30 * 24 * 60 * 60 * 1000)
      .toISOString().slice(0, 10);
    return p.date >= cutoff;
  }
  return true; // all_time
}

function setHTML(id, html) {
  const el = $(id);
  if (el) el.innerHTML = html;
}

function fmt(n, decimals = 1) {
  if (n == null || isNaN(n)) return '—';
  return Number(n).toFixed(decimals);
}

function fmtOdds(n) {
  if (n == null) return '—';
  return n > 0 ? `+${n}` : `${n}`;
}

function fmtProfit(n) {
  if (n == null) return '—';
  const s = (n >= 0 ? '+' : '') + fmt(n, 2) + 'u';
  return `<span class="${n >= 0 ? 'profit-pos' : 'profit-neg'}">${s}</span>`;
}

function statusBadge(status) {
  const cls = { Win: 'badge-win', Loss: 'badge-loss', Pending: 'badge-pending' }[status] ?? 'badge-pending';
  return `<span class="badge ${cls}">${status}</span>`;
}

function rowClass(status) {
  return { Win: 'row-win', Loss: 'row-loss', Pending: 'row-pending' }[status] ?? 'row-pending';
}

function valueClass(n) {
  if (n == null) return 'neutral';
  return n > 0 ? 'positive' : n < 0 ? 'negative' : 'neutral';
}

function confidenceBadge(n) {
  if (n == null) return '';
  const stars = '★'.repeat(n) + '☆'.repeat(5 - n);
  const cls   = n >= 4 ? 'conf-high' : n >= 3 ? 'conf-mid' : 'conf-low';
  return `<span class="badge ${cls}" title="${n}/5 confidence">${stars}</span>`;
}

function evBadge(ev) {
  if (ev == null) return '—';
  const pct = (ev * 100).toFixed(1);
  const cls = ev >= 0 ? 'ev-positive' : 'ev-negative';
  return `<span class="${cls}">${ev >= 0 ? '+' : ''}${pct}%</span>`;
}

// ── Model self-tuning status ───────────────────────────────────────────────
function renderModelStatus(model) {
  const el = $('model-status');
  if (!el || !model || model.blend_weight == null) return;
  const w = (model.blend_weight * 100).toFixed(0);
  let text;
  if (model.self_tuned) {
    text = `<span class="tuned">Self-tuned</span> — picks blend ${w}% market /
            ${100 - w}% model, learned from ${model.calibration_games.toLocaleString()}
            graded games and re-fit after every grading run.`;
  } else {
    const n = model.calibration_games ?? 0;
    text = `Self-tuning calibration is still collecting data (${n} graded games so far) —
            picks currently blend ${w}% market / ${100 - w}% model by default.`;
  }
  el.innerHTML = text;
  el.style.display = 'block';
}

// ── Last updated ───────────────────────────────────────────────────────────
function renderLastUpdated(iso) {
  if (!iso) return;
  const d = new Date(iso);
  document.getElementById('last-updated').textContent =
    'Updated ' + d.toLocaleString('en-US', {
      month: 'short', day: 'numeric',
      hour: 'numeric', minute: '2-digit', timeZoneName: 'short',
    });
}

// ── Stats computed from filtered history ──────────────────────────────────
function computeStats(history, targetMode) {
  const rows = history.filter(p => inTimeRange(p, targetMode));
  const graded  = rows.filter(p => p.status === 'Win' || p.status === 'Loss');
  const pending = rows.filter(p => p.status === 'Pending').length;
  const wins    = graded.filter(p => p.status === 'Win').length;
  const losses  = graded.filter(p => p.status === 'Loss').length;
  const wagered = graded.reduce((s, p) => s + (p.units ?? 0), 0);
  const profit  = graded.reduce((s, p) => s + (p.profit ?? 0), 0);
  return {
    wins, losses, pending,
    total_units_wagered: wagered,
    total_profit: profit,
    roi_pct:  wagered > 0 ? (profit / wagered) * 100 : 0,
    win_rate: graded.length > 0 ? (wins / graded.length) * 100 : 0,
  };
}

// ── Stats cards ────────────────────────────────────────────────────────────
function renderStats() {
  const rows  = marketFiltered(allHistory);
  const s     = computeStats(rows, mode);
  const otherMode  = mode === 'all_time' ? 'ytd' : 'all_time';
  const other      = computeStats(rows, otherMode);
  const otherLabel = otherMode === 'ytd' ? 'YTD' : 'All-time';

  const wins    = s.wins    ?? 0;
  const losses  = s.losses  ?? 0;
  const pending = s.pending ?? 0;
  const roi     = s.roi_pct ?? 0;
  const profit  = s.total_profit ?? 0;
  const winRate = s.win_rate ?? 0;

  setHTML('card-record',
    `<div class="value neutral">${wins}–${losses}</div>
     <div class="sub">${pending} pending · ${otherLabel}: ${other.wins}–${other.losses}</div>`);

  setHTML('card-winrate',
    `<div class="value ${valueClass(winRate - 50)}">${fmt(winRate)}%</div>
     <div class="sub">${otherLabel}: ${fmt(other.win_rate)}%</div>`);

  setHTML('card-roi',
    `<div class="value ${valueClass(roi)}">${roi >= 0 ? '+' : ''}${fmt(roi)}%</div>
     <div class="sub">${otherLabel}: ${other.roi_pct >= 0 ? '+' : ''}${fmt(other.roi_pct)}%</div>`);

  setHTML('card-profit',
    `<div class="value ${valueClass(profit)}">${profit >= 0 ? '+' : ''}${fmt(profit, 2)}u</div>
     <div class="sub">${fmt(s.total_units_wagered, 1)}u wagered · ${otherLabel}: ${other.total_profit >= 0 ? '+' : ''}${fmt(other.total_profit, 2)}u</div>`);
}

// ── Slate helpers (presentational sort + tier filter) ──────────────────────
// Apply the active sort + tier filter for a slate. Pure reordering/filtering —
// no values are recomputed, so the numbers rendered are identical to source.
function applySlate(list, st) {
  let rows = (list || []).slice();
  if (st.tier === 'high')     rows = rows.filter(p => (p.confidence ?? 0) >= 4);
  else if (st.tier === 'mid') rows = rows.filter(p => Math.round(p.confidence ?? 0) === 3);
  const key = st.sort;
  rows.sort((a, b) => (b[key] ?? -Infinity) - (a[key] ?? -Infinity));
  return rows;
}

// Composed empty state with an inline icon.
function emptyState(title, desc) {
  return `
    <div class="empty-state">
      <svg class="empty-icon" width="30" height="30" viewBox="0 0 24 24" fill="none" aria-hidden="true">
        <rect x="3" y="5" width="18" height="15" rx="2" stroke="currentColor" stroke-width="1.5"/>
        <path d="M3 9h18M8 3v4M16 3v4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
      </svg>
      <div class="empty-title">${title}</div>
      <div class="empty-desc">${desc}</div>
    </div>`;
}

// Model-vs-market probability bars rendered from existing fields only.
function probDelta(modelProb, impliedProb) {
  if (modelProb == null || impliedProb == null) return '';
  const m = Math.max(0, Math.min(100, modelProb * 100));
  const k = Math.max(0, Math.min(100, impliedProb * 100));
  return `
    <div class="prob-delta" aria-label="Model ${fmt(m)} percent versus market ${fmt(k)} percent">
      <div class="prob-row">
        <span class="prob-name">Model</span>
        <span class="prob-bar model"><i style="width:${m}%"></i></span>
        <span class="prob-val">${fmt(m)}%</span>
      </div>
      <div class="prob-row">
        <span class="prob-name">Market</span>
        <span class="prob-bar market"><i style="width:${k}%"></i></span>
        <span class="prob-val">${fmt(k)}%</span>
      </div>
    </div>`;
}

function edgeMetric(edge) {
  if (edge == null) return `<span class="metric-value">—</span>`;
  const cls = edge >= 0 ? 'ev-positive' : 'ev-negative';
  return `<span class="metric-value ${cls}">${edge >= 0 ? '+' : ''}${fmt(edge * 100)}%</span>`;
}

function pitchersLine(p) {
  return (p.away_pitcher || p.home_pitcher)
    ? `<span class="pick-pitchers">${p.away_pitcher || 'TBD'} vs ${p.home_pitcher || 'TBD'}</span>`
    : '';
}

// ── Today's moneyline picks ────────────────────────────────────────────────
function renderTodayPicks() {
  if (!todayPicks || todayPicks.length === 0) {
    setHTML('today-picks', emptyState('No moneyline slate locked yet',
      'Picks post after the morning model run. Check back later today.'));
    return;
  }
  const rows = applySlate(todayPicks, slateState.picks);
  if (rows.length === 0) {
    setHTML('today-picks', '<p class="picks-empty">No picks match this tier filter.</p>');
    return;
  }

  setHTML('today-picks', rows.map(p => {
    const game = `${p.away_team} @ ${p.home_team}`;
    const raw  = p.raw_model_prob != null
      ? `<span class="pick-meta">Raw model ${fmt(p.raw_model_prob * 100)}% → ${fmt(p.model_prob * 100)}% blended</span>`
      : '';
    const pred = (p.predicted_home_runs != null && p.predicted_away_runs != null)
      ? `<span class="pick-meta score-pred">Pred ${fmt(p.predicted_away_runs)} – ${fmt(p.predicted_home_runs)}</span>`
      : '';
    return `
      <div class="pick-card">
        <span class="game-label">${game}</span>
        <div class="pick-badges">${confidenceBadge(p.confidence)}${statusBadge(p.status)}</div>
        <div class="pick-headline">
          <span class="pick-team">${p.pick}</span>
          <span class="pick-odds">${fmtOdds(p.odds)}</span>
          <span class="pick-units">${p.units}u</span>
        </div>
        <div class="pick-metrics">
          <div class="metric"><span class="metric-label">EV</span><span class="metric-value">${evBadge(p.ev)}</span></div>
          <div class="metric"><span class="metric-label">Edge</span>${edgeMetric(p.edge)}</div>
        </div>
        ${probDelta(p.model_prob, p.implied_prob)}
        <div class="pick-foot">${pitchersLine(p)}${pred}${raw}</div>
      </div>`;
  }).join(''));
}

// ── Today's totals (Over/Under) picks ──────────────────────────────────────
function renderTodayTotals() {
  if (!todayTotals || todayTotals.length === 0) {
    setHTML('today-totals', emptyState('No totals slate locked yet',
      'Over/under picks post after the morning model run. Check back later today.'));
    return;
  }
  const rows = applySlate(todayTotals, slateState.totals);
  if (rows.length === 0) {
    setHTML('today-totals', '<p class="picks-empty">No picks match this tier filter.</p>');
    return;
  }

  setHTML('today-totals', rows.map(p => {
    const game = `${p.away_team} @ ${p.home_team}`;
    const line = p.listed_total != null ? ` ${fmt(p.listed_total)}` : '';
    const pred = p.predicted_total != null
      ? `<span class="pick-meta score-pred">Model total ${fmt(p.predicted_total)}</span>`
      : '';
    return `
      <div class="pick-card">
        <span class="game-label">${game}</span>
        <div class="pick-badges">${confidenceBadge(p.confidence)}${statusBadge(p.status)}</div>
        <div class="pick-headline">
          <span class="pick-team">${p.pick}<span class="pick-line">${line}</span></span>
          <span class="pick-odds">${fmtOdds(p.odds)}</span>
          <span class="pick-units">${p.units}u</span>
        </div>
        <div class="pick-metrics">
          <div class="metric"><span class="metric-label">EV</span><span class="metric-value">${evBadge(p.ev)}</span></div>
          <div class="metric"><span class="metric-label">Edge</span>${edgeMetric(p.edge)}</div>
        </div>
        ${probDelta(p.model_prob, p.implied_prob)}
        <div class="pick-foot">${pitchersLine(p)}${pred}</div>
      </div>`;
  }).join(''));
}

// ── Cumulative P&L chart ───────────────────────────────────────────────────
function renderChart(history, currentMode) {
  let graded = history.filter(p =>
    (p.status === 'Win' || p.status === 'Loss') && inTimeRange(p, currentMode));
  graded = [...graded].sort((a, b) => a.date.localeCompare(b.date));

  let cumulative = 0;
  const labels = [];
  const values = [];
  graded.forEach((p, i) => {
    cumulative += p.profit ?? 0;
    labels.push(
      i === 0 || i === graded.length - 1 ||
      i % Math.max(1, Math.floor(graded.length / 20)) === 0
        ? p.date.slice(5)
        : ''
    );
    values.push(parseFloat(cumulative.toFixed(2)));
  });

  const ctx = document.getElementById('pnl-chart').getContext('2d');
  if (chart) chart.destroy();

  const finalValue = values[values.length - 1] ?? 0;
  const lineColor  = finalValue >= 0 ? '#2f6b3a' : '#a3322f';
  const monoFont   = "'JetBrains Mono', ui-monospace, monospace";

  chart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        data: values,
        borderColor: lineColor,
        borderWidth: 2,
        pointRadius: 0,
        pointHoverRadius: 4,
        pointHoverBackgroundColor: lineColor,
        pointHoverBorderColor: '#ffffff',
        pointHoverBorderWidth: 2,
        fill: true,
        backgroundColor: (ctx) => {
          const gradient = ctx.chart.ctx.createLinearGradient(0, 0, 0, 240);
          gradient.addColorStop(0, lineColor + '22');
          gradient.addColorStop(1, lineColor + '00');
          return gradient;
        },
        tension: 0.3,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#ffffff',
          borderColor: '#e7e4dd',
          borderWidth: 1,
          titleColor: '#8a877e',
          bodyColor: '#23211c',
          bodyFont: { family: monoFont },
          padding: 10,
          cornerRadius: 8,
          displayColors: false,
          callbacks: {
            label: (ctx) => {
              const v = ctx.parsed.y;
              return (v >= 0 ? '+' : '') + v.toFixed(2) + ' units';
            },
          },
        },
      },
      scales: {
        x: {
          ticks: { color: '#8a877e', maxRotation: 0, font: { family: monoFont, size: 10 } },
          grid:  { color: 'rgba(35,33,28,.06)' },
          border: { color: '#e7e4dd' },
        },
        y: {
          ticks: {
            color: '#8a877e',
            font: { family: monoFont, size: 10 },
            callback: (v) => (v >= 0 ? '+' : '') + v.toFixed(1) + 'u',
          },
          grid: { color: 'rgba(35,33,28,.06)' },
          border: { display: false },
        },
      },
    },
  });
}

// ── History table ──────────────────────────────────────────────────────────
function renderTable(history) {
  let rows = marketFiltered(history).filter(p => inTimeRange(p, mode));

  if (filterText) {
    const q = filterText.toLowerCase();
    rows = rows.filter(p =>
      (p.away_team ?? '').toLowerCase().includes(q) ||
      (p.home_team ?? '').toLowerCase().includes(q) ||
      (p.pick      ?? '').toLowerCase().includes(q) ||
      (p.status    ?? '').toLowerCase().includes(q) ||
      (p.bet_type  ?? '').toLowerCase().includes(q)
    );
  }

  if (rows.length === 0) {
    setHTML('history-tbody', `<tr><td colspan="10" style="text-align:center;color:var(--muted);padding:30px">No picks found.</td></tr>`);
    return;
  }

  setHTML('history-tbody', rows.map(p => {
    const game = `${p.away_team} @ ${p.home_team}`;
    const pickLabel = (p.bet_type === 'totals' && p.listed_total != null)
      ? `${p.pick} ${fmt(p.listed_total)}`
      : (p.pick ?? '—');
    return `
      <tr class="${rowClass(p.status)}">
        <td>${p.date ?? '—'}</td>
        <td>${game}</td>
        <td style="font-weight:600">${pickLabel}</td>
        <td>${fmtOdds(p.odds)}</td>
        <td>${p.units ?? '—'}u</td>
        <td>${p.model_prob != null ? fmt(p.model_prob * 100) + '%' : '—'}</td>
        <td>${p.edge != null ? '+' + fmt(p.edge * 100) + '%' : '—'}</td>
        <td>${confidenceBadge(p.confidence)}</td>
        <td>${statusBadge(p.status)}</td>
        <td>${p.profit != null ? fmtProfit(p.profit) : '—'}</td>
      </tr>`;
  }).join(''));
}

// ── Pick of the Day ────────────────────────────────────────────────────────
function renderPickOfDay(picks) {
  if (!picks || picks.length === 0) {
    setHTML('potd-card', '<p style="color:var(--muted)">No picks for today yet — check back after the morning run.</p>');
    return;
  }
  const top = picks.reduce((best, p) => ((p.edge ?? 0) > (best.edge ?? 0) ? p : best), picks[0]);
  const line = (top.bet_type === 'totals' && top.listed_total != null) ? ` ${fmt(top.listed_total)}` : '';
  setHTML('potd-card', `
    <div class="potd-label">Best edge today</div>
    <div class="potd-game">${top.away_team} @ ${top.home_team}</div>
    <div class="potd-pick">${top.pick}${line}</div>
    <div class="potd-meta">
      <div class="metric"><span class="metric-label">Edge</span><span class="metric-value positive">+${fmt(top.edge * 100)}%</span></div>
      <div class="metric"><span class="metric-label">EV</span><span class="metric-value">${evBadge(top.ev)}</span></div>
      <div class="metric"><span class="metric-label">Odds</span><span class="metric-value">${fmtOdds(top.odds)}</span></div>
      <div class="metric"><span class="metric-label">Model</span><span class="metric-value">${fmt(top.model_prob * 100)}%</span></div>
      <div class="metric"><span class="metric-label">Stake</span><span class="metric-value">${top.units}u</span></div>
    </div>
    <div style="margin-top:14px">${confidenceBadge(top.confidence)}</div>`);
}


// ── Error banner ───────────────────────────────────────────────────────────
function showError(msg) {
  const banner = $('error-banner');
  const text   = $('error-text');
  if (banner) banner.style.display = 'block';
  if (text)   text.textContent = msg;
}

// ── Toggles (time range + market) ──────────────────────────────────────────
function rerenderFiltered() {
  renderStats();
  renderChart(marketFiltered(allHistory), mode);
  renderTable(allHistory);
}

function setMode(newMode) {
  mode = newMode;
  document.querySelectorAll('.toggle-btn[data-mode]').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.mode === mode);
  });
  rerenderFiltered();
}

function setMarket(newMarket) {
  market = newMarket;
  document.querySelectorAll('.toggle-btn[data-market]').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.market === market);
  });
  rerenderFiltered();
}

// ── Init ───────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.toggle-btn[data-mode]').forEach(btn => {
    btn.addEventListener('click', () => setMode(btn.dataset.mode));
  });
  document.querySelectorAll('.toggle-btn[data-market]').forEach(btn => {
    btn.addEventListener('click', () => setMarket(btn.dataset.market));
  });

  // Today's-slate sort + tier controls (presentational re-render only)
  document.querySelectorAll('.slate-controls').forEach(group => {
    const kind   = group.dataset.slate;                 // 'picks' | 'totals'
    const render = kind === 'totals' ? renderTodayTotals : renderTodayPicks;
    group.querySelectorAll('.toggle-btn[data-sort]').forEach(btn => {
      btn.addEventListener('click', () => {
        slateState[kind].sort = btn.dataset.sort;
        group.querySelectorAll('.toggle-btn[data-sort]')
          .forEach(b => b.classList.toggle('active', b === btn));
        render();
      });
    });
    group.querySelectorAll('.toggle-btn[data-tier]').forEach(btn => {
      btn.addEventListener('click', () => {
        slateState[kind].tier = btn.dataset.tier;
        group.querySelectorAll('.toggle-btn[data-tier]')
          .forEach(b => b.classList.toggle('active', b === btn));
        render();
      });
    });
  });

  const searchInput = document.getElementById('table-search');
  if (searchInput) {
    searchInput.addEventListener('input', (e) => {
      filterText = e.target.value.trim();
      renderTable(allHistory);
    });
  }

  loadData();
});
