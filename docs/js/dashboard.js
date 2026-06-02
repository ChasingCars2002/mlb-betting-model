/* dashboard.js — BaseballBetBot GitHub Pages dashboard */
'use strict';

// ── State ──────────────────────────────────────────────────────────────────
let allHistory  = [];
let statsData   = {};
let chart       = null;
let mode        = 'ytd';    // 'ytd' | 'all_time'
let filterText  = '';

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
    const rawHistory = await historyRes.json();
    allHistory = rawHistory.filter(p => !p.bet_type || p.bet_type === 'moneyline');
    const rawToday = picksRes.ok ? await picksRes.json() : [];
    const rawTotals = totalsRes && totalsRes.ok ? await totalsRes.json() : [];
    // Only show picks that are actually for today (US Eastern). Between runs the
    // committed picks_today.json still holds the previous day's picks; without
    // this guard the site would display yesterday's matchups as "Today's Picks".
    const today = easternDateStr();
    const todayPicks = rawToday.filter(p => p.date === today);
    const todayTotals = rawTotals.filter(p => p.date === today);

    const loadingEl = document.getElementById('loading');
    const appEl     = document.getElementById('app');
    if (loadingEl) loadingEl.style.display = 'none';
    if (appEl)     appEl.style.display     = 'block';

    renderLastUpdated(statsData.last_updated);
    renderStats();
    renderChart(allHistory, mode);
    renderTable(allHistory);
    renderPickOfDay(todayPicks);
    renderTodayPicks(todayPicks);
    renderTodayTotals(todayTotals);

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

// ── Stats computed from filtered history (moneyline only) ─────────────────
function computeStats(history, targetMode) {
  const ytdYear = String(new Date().getFullYear());
  const rows = targetMode === 'ytd'
    ? history.filter(p => p.date && p.date.startsWith(ytdYear))
    : history;
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
  const s     = computeStats(allHistory, mode);
  const other = computeStats(allHistory, mode === 'ytd' ? 'all_time' : 'ytd');
  const otherLabel = mode === 'ytd' ? 'All-time' : 'YTD';

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

// ── Today's moneyline picks ────────────────────────────────────────────────
function renderTodayPicks(picks) {
  if (!picks || picks.length === 0) {
    setHTML('today-picks', '<p class="picks-empty">No moneyline picks for today yet — check back after the morning run.</p>');
    return;
  }

  setHTML('today-picks', picks.map(p => {
    const game  = `${p.away_team} @ ${p.home_team}`;
    const edge  = p.edge != null ? `Edge: +${fmt(p.edge * 100)}%` : '';
    const ev    = evBadge(p.ev);
    const conf  = confidenceBadge(p.confidence);
    const pred  = (p.predicted_home_runs != null && p.predicted_away_runs != null)
      ? `<span class="pick-meta score-pred">Pred: ${fmt(p.predicted_away_runs)} – ${fmt(p.predicted_home_runs)}</span>`
      : '';
    return `
      <div class="pick-card">
        <span class="game-label">${game}</span>
        <span class="pick-team">${p.pick}</span>
        <span class="pick-meta">${fmtOdds(p.odds)} · ${p.units}u</span>
        <span class="pick-meta">${edge} · EV: ${ev}</span>
        <span class="pick-meta">Model: ${fmt(p.model_prob * 100)}% · Implied: ${fmt(p.implied_prob * 100)}%</span>
        ${pred}
        <div class="pick-badges">${conf}${statusBadge(p.status)}</div>
      </div>`;
  }).join(''));
}

// ── Today's totals (Over/Under) picks ──────────────────────────────────────
function renderTodayTotals(picks) {
  if (!picks || picks.length === 0) {
    setHTML('today-totals', '<p class="picks-empty">No Over/Under picks for today yet — check back after the morning run.</p>');
    return;
  }

  setHTML('today-totals', picks.map(p => {
    const game = `${p.away_team} @ ${p.home_team}`;
    const edge = p.edge != null ? `Edge: +${fmt(p.edge * 100)}%` : '';
    const ev   = evBadge(p.ev);
    const conf = confidenceBadge(p.confidence);
    const line = p.listed_total != null ? ` ${fmt(p.listed_total)}` : '';
    const model = p.predicted_total != null
      ? `<span class="pick-meta score-pred">Model total: ${fmt(p.predicted_total)}</span>`
      : '';
    return `
      <div class="pick-card">
        <span class="game-label">${game}</span>
        <span class="pick-team">${p.pick}${line}</span>
        <span class="pick-meta">${fmtOdds(p.odds)} · ${p.units}u</span>
        <span class="pick-meta">${edge} · EV: ${ev}</span>
        <span class="pick-meta">Model: ${fmt(p.model_prob * 100)}% · Implied: ${fmt(p.implied_prob * 100)}%</span>
        ${model}
        <div class="pick-badges">${conf}${statusBadge(p.status)}</div>
      </div>`;
  }).join(''));
}

// ── Cumulative P&L chart ───────────────────────────────────────────────────
function renderChart(history, currentMode) {
  const ytdYear = new Date().getFullYear();

  let graded = history.filter(p => p.status === 'Win' || p.status === 'Loss');
  if (currentMode === 'ytd') {
    graded = graded.filter(p => p.date && p.date.startsWith(String(ytdYear)));
  }
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
  const lineColor  = finalValue >= 0 ? '#3fb950' : '#f85149';

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
        fill: true,
        backgroundColor: (ctx) => {
          const gradient = ctx.chart.ctx.createLinearGradient(0, 0, 0, 220);
          gradient.addColorStop(0, lineColor + '33');
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
          backgroundColor: '#161b22',
          borderColor: '#30363d',
          borderWidth: 1,
          titleColor: '#8b949e',
          bodyColor: '#e6edf3',
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
          ticks: { color: '#8b949e', maxRotation: 0, font: { size: 11 } },
          grid:  { color: '#30363d' },
        },
        y: {
          ticks: {
            color: '#8b949e',
            font: { size: 11 },
            callback: (v) => (v >= 0 ? '+' : '') + v.toFixed(1) + 'u',
          },
          grid: { color: '#30363d' },
        },
      },
    },
  });
}

// ── History table ──────────────────────────────────────────────────────────
function renderTable(history) {
  const ytdYear = String(new Date().getFullYear());
  let rows = history;

  if (mode === 'ytd') {
    rows = rows.filter(p => p.date && p.date.startsWith(ytdYear));
  }

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
    return `
      <tr class="${rowClass(p.status)}">
        <td>${p.date ?? '—'}</td>
        <td>${game}</td>
        <td style="font-weight:600">${p.pick ?? '—'}</td>
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
  const conf = confidenceBadge(top.confidence);
  setHTML('potd-card', `
    <div class="potd-label">★ Best Edge Today</div>
    <div class="potd-game">${top.away_team} @ ${top.home_team}</div>
    <div class="potd-pick">${top.pick}</div>
    <div class="potd-meta">
      <span>Odds: ${fmtOdds(top.odds)}</span>
      <span>Edge: +${fmt(top.edge * 100)}%</span>
      <span>Model: ${fmt(top.model_prob * 100)}%</span>
      <span>Units: ${top.units}u</span>
    </div>
    <div style="margin-top:8px">${conf}</div>`);
}


// ── Error banner ───────────────────────────────────────────────────────────
function showError(msg) {
  const banner = $('error-banner');
  const text   = $('error-text');
  if (banner) banner.style.display = 'block';
  if (text)   text.textContent = msg;
}

// ── Toggle (YTD / All-Time) ────────────────────────────────────────────────
function setMode(newMode) {
  mode = newMode;
  document.querySelectorAll('.toggle-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.mode === mode);
  });
  renderStats();
  renderChart(allHistory, mode);
  renderTable(allHistory);
}

// ── Init ───────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.toggle-btn').forEach(btn => {
    btn.addEventListener('click', () => setMode(btn.dataset.mode));
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
