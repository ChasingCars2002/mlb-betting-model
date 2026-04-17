/* dashboard.js — BaseballBetBot GitHub Pages dashboard */
'use strict';

// ── State ──────────────────────────────────────────────────────────────────
let allHistory = [];
let statsData  = {};
let chart      = null;
let mode       = 'ytd'; // 'ytd' | 'all_time'
let filterText = '';

// ── Fetch ──────────────────────────────────────────────────────────────────
async function loadData() {
  try {
    const [statsRes, historyRes, todayRes] = await Promise.all([
      fetch('data/stats.json'),
      fetch('data/picks_history.json'),
      fetch('data/picks_today.json'),
    ]);

    if (!statsRes.ok || !historyRes.ok || !todayRes.ok) {
      throw new Error('One or more data files could not be loaded.');
    }

    statsData  = await statsRes.json();
    allHistory = await historyRes.json();
    const todayPicks = await todayRes.json();

    document.getElementById('loading').style.display = 'none';
    document.getElementById('app').style.display = 'block';

    renderLastUpdated(statsData.last_updated);
    renderStats(statsData[mode]);
    renderPickOfDay(todayPicks);
    renderTodayPicks(todayPicks);
    renderChart(allHistory, mode);
    renderTable(allHistory);

  } catch (err) {
    document.getElementById('loading').style.display = 'none';
    showError('Could not load data. ' + err.message +
      ' If viewing locally, serve with a local HTTP server (e.g. python -m http.server).');
  }
}

// ── Helpers ────────────────────────────────────────────────────────────────
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

// ── Last updated ───────────────────────────────────────────────────────────
function renderLastUpdated(iso) {
  if (!iso) return;
  const d = new Date(iso);
  document.getElementById('last-updated').textContent =
    'Updated ' + d.toLocaleString('en-US', { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit', timeZoneName: 'short' });
}

// ── Stats cards ────────────────────────────────────────────────────────────
function renderStats(s) {
  if (!s) return;

  const wins    = s.wins    ?? 0;
  const losses  = s.losses  ?? 0;
  const pending = s.pending ?? 0;
  const roi     = s.roi_pct ?? 0;
  const profit  = s.total_profit ?? 0;
  const winRate = s.win_rate ?? 0;

  const other = mode === 'ytd' ? statsData.all_time : statsData.ytd;
  const otherLabel = mode === 'ytd' ? 'All-time' : 'YTD';

  document.getElementById('card-record').innerHTML =
    `<div class="value neutral">${wins}–${losses}</div>
     <div class="sub">${pending} pending · ${otherLabel}: ${other?.wins ?? 0}–${other?.losses ?? 0}</div>`;

  document.getElementById('card-winrate').innerHTML =
    `<div class="value ${valueClass(winRate - 50)}">${fmt(winRate)}%</div>
     <div class="sub">${otherLabel}: ${fmt(other?.win_rate)}%</div>`;

  document.getElementById('card-roi').innerHTML =
    `<div class="value ${valueClass(roi)}">${roi >= 0 ? '+' : ''}${fmt(roi)}%</div>
     <div class="sub">${otherLabel}: ${other?.roi_pct >= 0 ? '+' : ''}${fmt(other?.roi_pct)}%</div>`;

  document.getElementById('card-profit').innerHTML =
    `<div class="value ${valueClass(profit)}">${profit >= 0 ? '+' : ''}${fmt(profit, 2)}u</div>
     <div class="sub">${fmt(s.total_units_wagered, 1)}u wagered · ${otherLabel}: ${other?.total_profit >= 0 ? '+' : ''}${fmt(other?.total_profit, 2)}u</div>`;
}

// ── Today's picks ──────────────────────────────────────────────────────────
function renderTodayPicks(picks) {
  const el = document.getElementById('today-picks');
  if (!picks || picks.length === 0) {
    el.innerHTML = '<p class="picks-empty">No picks for today yet — check back after the morning run.</p>';
    return;
  }

  el.innerHTML = picks.map(p => {
    const game = `${p.away_team} @ ${p.home_team}`;
    const edge = p.edge != null ? `Edge: +${fmt(p.edge * 100)}%` : '';
    const ev   = p.ev   != null ? `EV: +${fmt(p.ev   * 100)}%` : '';
    return `
      <div class="pick-card">
        <span class="game-label">${game}</span>
        <span class="pick-team">${p.pick}</span>
        <span class="pick-meta">${fmtOdds(p.odds)} · ${p.units}u</span>
        <span class="pick-meta">${edge} · ${ev}</span>
        <span class="pick-meta">Model: ${fmt(p.model_prob * 100)}% · Implied: ${fmt(p.implied_prob * 100)}%</span>
        ${statusBadge(p.status)}
      </div>`;
  }).join('');
}

// ── Cumulative P&L chart ───────────────────────────────────────────────────
function renderChart(history, currentMode) {
  const ytdYear = new Date().getFullYear();

  // Only graded bets; optionally filter to current year for YTD
  let graded = history.filter(p => p.status === 'Win' || p.status === 'Loss');
  if (currentMode === 'ytd') {
    graded = graded.filter(p => p.date && p.date.startsWith(String(ytdYear)));
  }
  // Sort oldest → newest for cumulative sum
  graded = [...graded].sort((a, b) => a.date.localeCompare(b.date));

  let cumulative = 0;
  const labels = [];
  const values = [];
  graded.forEach((p, i) => {
    cumulative += p.profit ?? 0;
    // Label every ~10th point or first/last to keep x-axis readable
    labels.push(i === 0 || i === graded.length - 1 || i % Math.max(1, Math.floor(graded.length / 20)) === 0
      ? p.date.slice(5) // MM-DD
      : '');
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
      (p.away_team  ?? '').toLowerCase().includes(q) ||
      (p.home_team  ?? '').toLowerCase().includes(q) ||
      (p.pick       ?? '').toLowerCase().includes(q) ||
      (p.status     ?? '').toLowerCase().includes(q)
    );
  }

  const tbody = document.getElementById('history-tbody');
  if (rows.length === 0) {
    tbody.innerHTML = `<tr><td colspan="9" style="text-align:center;color:var(--muted);padding:30px">No picks found.</td></tr>`;
    return;
  }

  tbody.innerHTML = rows.map(p => {
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
        <td>${statusBadge(p.status)}</td>
        <td>${p.profit != null ? fmtProfit(p.profit) : '—'}</td>
      </tr>`;
  }).join('');
}

// ── Pick of the Day ────────────────────────────────────────────────────────
function renderPickOfDay(picks) {
  const el = document.getElementById('potd-card');
  if (!picks || picks.length === 0) {
    el.innerHTML = '<p style="color:var(--muted)">No picks for today yet \u2014 check back after the morning run.</p>';
    return;
  }
  const top = picks.reduce((best, p) => ((p.edge ?? 0) > (best.edge ?? 0) ? p : best), picks[0]);
  el.innerHTML = `
    <div class="potd-label">\u2605 Best Edge Today</div>
    <div class="potd-game">${top.away_team} @ ${top.home_team}</div>
    <div class="potd-pick">${top.pick}</div>
    <div class="potd-meta">
      <span>Odds: ${fmtOdds(top.odds)}</span>
      <span>Edge: +${fmt(top.edge * 100)}%</span>
      <span>Model: ${fmt(top.model_prob * 100)}%</span>
      <span>Units: ${top.units}u</span>
    </div>`;
}

// ── Bitcoin tip jar ────────────────────────────────────────────────────────
function copyBtc() {
  const addr = 'bc1q9kwf5fc35ruuuvcpe8j0zsm856c6dxr7k4887n';
  const done = () => showBtcToast('Bitcoin address copied!');
  if (navigator.clipboard) {
    navigator.clipboard.writeText(addr).then(done).catch(() => fallbackCopy(addr, done));
  } else {
    fallbackCopy(addr, done);
  }
}

function fallbackCopy(text, cb) {
  const ta = document.createElement('textarea');
  ta.value = text;
  ta.style.position = 'fixed';
  ta.style.opacity = '0';
  document.body.appendChild(ta);
  ta.select();
  document.execCommand('copy');
  document.body.removeChild(ta);
  cb();
}

function showBtcToast(msg) {
  let toast = document.getElementById('btc-toast');
  if (!toast) {
    toast = document.createElement('div');
    toast.id = 'btc-toast';
    toast.className = 'btc-toast';
    document.body.appendChild(toast);
  }
  toast.textContent = msg;
  toast.classList.add('show');
  setTimeout(() => toast.classList.remove('show'), 2200);
}

// ── Error banner ───────────────────────────────────────────────────────────
function showError(msg) {
  document.getElementById('error-banner').style.display = 'block';
  document.getElementById('error-text').textContent = msg;
}

// ── Toggle (YTD / All-Time) ────────────────────────────────────────────────
function setMode(newMode) {
  mode = newMode;
  document.querySelectorAll('.toggle-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.mode === mode);
  });
  renderStats(statsData[mode]);
  renderChart(allHistory, mode);
  renderTable(allHistory);
}

// ── Init ───────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  // Toggle buttons
  document.querySelectorAll('.toggle-btn').forEach(btn => {
    btn.addEventListener('click', () => setMode(btn.dataset.mode));
  });

  // Search filter
  const searchInput = document.getElementById('table-search');
  if (searchInput) {
    searchInput.addEventListener('input', (e) => {
      filterText = e.target.value.trim();
      renderTable(allHistory);
    });
  }

  loadData();
});
