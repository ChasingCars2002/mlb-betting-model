/* auth.js — Supabase Auth + Stripe subscription management */
'use strict';

const SUPABASE_URL = 'https://hmukgvrpuncxkzeuujam.supabase.co';
const SUPABASE_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImhtdWtndnJwdW5jeGt6ZXV1amFtIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzgxOTMwNDgsImV4cCI6MjA5Mzc2OTA0OH0.0SC2M6Fro3dOwWwf-7Qld-aN3RjCusNkKUWVEalLtEM';
const PICKS_FN    = `${SUPABASE_URL}/functions/v1/get-picks-data`;
const CHECKOUT_FN = `${SUPABASE_URL}/functions/v1/create-checkout-session`;

// Initialized inside DOMContentLoaded once we confirm the SDK loaded
let _sb = null;

// ── Shared auth state (read by dashboard.js) ───────────────────────────────
window.sbUser       = null;
window.sbSession    = null;
window.sbSubscribed = false;

// ── Modal (defined immediately so onclick handlers work even before SDK) ───
window.openModal = function (tab) {
  const m = document.getElementById('auth-modal');
  if (!m) return;
  m.style.display = 'flex';
  _tab(tab || 'signin');
  _clearErr();
};

window.closeModal = function () {
  const m = document.getElementById('auth-modal');
  if (m) m.style.display = 'none';
};

function _tab(name) {
  document.querySelectorAll('.modal-tab').forEach(t =>
    t.classList.toggle('active', t.dataset.tab === name));
  document.querySelectorAll('.modal-panel').forEach(p => {
    p.style.display = (p.dataset.panel === name) ? 'flex' : 'none';
  });
}

function _err(msg) {
  document.querySelectorAll('.modal-error').forEach(el => {
    el.textContent = msg;
    el.style.display = msg ? 'block' : 'none';
  });
}
function _clearErr() { _err(''); }

// ── Header auth bar ────────────────────────────────────────────────────────
function _updateHeader() {
  const loggedOut = document.getElementById('auth-logged-out');
  const loggedIn  = document.getElementById('auth-logged-in');
  const emailEl   = document.getElementById('auth-user-email');
  if (!loggedOut || !loggedIn) return;
  if (window.sbUser) {
    loggedOut.style.display = 'none';
    loggedIn.style.display  = 'flex';
    if (emailEl) emailEl.textContent = window.sbUser.email;
  } else {
    loggedOut.style.display = 'flex';
    loggedIn.style.display  = 'none';
  }
}

// ── Auth actions (override the inline stubs in index.html) ────────────────
window.sbSignOut = async function () {
  if (_sb) await _sb.auth.signOut();
};

// ── Subscription check ─────────────────────────────────────────────────────
async function _checkSub() {
  if (!window.sbUser || !_sb) { window.sbSubscribed = false; return; }
  const { data } = await _sb
    .from('profiles')
    .select('subscription_status')
    .eq('user_id', window.sbUser.id)
    .single();
  window.sbSubscribed = data?.subscription_status === 'active';
}

// ── Authenticated fetch for gated picks data ───────────────────────────────
window.fetchGatedData = async function (file) {
  if (!window.sbSession) return null;
  try {
    const res = await fetch(`${PICKS_FN}?file=${file}`, {
      headers: { Authorization: `Bearer ${window.sbSession.access_token}` },
    });
    if (res.status === 402) return { __gated: true };
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
};

// ── Stripe checkout (overrides the inline stub in index.html) ─────────────
window.startSubscription = async function () {
  if (!window.sbUser) { openModal('signup'); return; }
  const btns = document.querySelectorAll('.subscribe-cta');
  btns.forEach(b => { b.disabled = true; b.textContent = 'Redirecting…'; });
  try {
    const res = await fetch(CHECKOUT_FN, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${window.sbSession.access_token}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ user_id: window.sbUser.id, email: window.sbUser.email }),
    });
    if (res.ok) {
      const { url } = await res.json();
      window.location.href = url;
    } else {
      alert('Could not start checkout. Please try again.');
      btns.forEach(b => { b.disabled = false; b.textContent = 'Subscribe — $7.99/mo'; });
    }
  } catch {
    alert('Network error. Please try again.');
    btns.forEach(b => { b.disabled = false; b.textContent = 'Subscribe — $7.99/mo'; });
  }
};

// ── Form submissions ───────────────────────────────────────────────────────
async function _onSignIn(e) {
  e.preventDefault();
  if (!_sb) return;
  _clearErr();
  const email = document.getElementById('si-email').value.trim();
  const pw    = document.getElementById('si-pw').value;
  const btn   = document.getElementById('si-btn');
  btn.disabled = true; btn.textContent = 'Signing in…';
  try {
    const { error } = await _sb.auth.signInWithPassword({ email, password: pw });
    if (error) throw error;
    closeModal();
  } catch (err) {
    _err(err.message);
    btn.disabled = false; btn.textContent = 'Sign in';
  }
}

async function _onSignUp(e) {
  e.preventDefault();
  if (!_sb) return;
  _clearErr();
  const email = document.getElementById('su-email').value.trim();
  const pw    = document.getElementById('su-pw').value;
  const btn   = document.getElementById('su-btn');
  btn.disabled = true; btn.textContent = 'Creating account…';
  try {
    const { error } = await _sb.auth.signUp({ email, password: pw });
    if (error) throw error;
    document.getElementById('signup-panel').innerHTML = `
      <div class="modal-success">
        <div style="font-size:2rem;margin-bottom:10px">📧</div>
        <div class="success-title">Check your email</div>
        <div class="success-body">
          A confirmation link was sent to <strong>${email}</strong>.<br>
          Click it to activate your account, then sign in here.
        </div>
      </div>`;
  } catch (err) {
    _err(err.message);
    btn.disabled = false; btn.textContent = 'Create account';
  }
}

// ── Auth init ──────────────────────────────────────────────────────────────
async function _initAuth() {
  const { data: { session } } = await _sb.auth.getSession();
  window.sbSession = session;
  window.sbUser    = session?.user ?? null;
  if (window.sbUser) await _checkSub();
  _updateHeader();

  _sb.auth.onAuthStateChange(async (_event, session) => {
    window.sbSession    = session;
    window.sbUser       = session?.user ?? null;
    window.sbSubscribed = false;
    if (window.sbUser) await _checkSub();
    _updateHeader();
    if (typeof window.onAuthChanged === 'function') {
      window.onAuthChanged(window.sbUser, window.sbSubscribed);
    }
  });

  // Return from Stripe checkout
  const params = new URLSearchParams(window.location.search);
  if (params.get('subscribed') === 'true') {
    window.history.replaceState({}, '', window.location.pathname);
    if (window.sbUser) {
      await _checkSub();
      _updateHeader();
      if (typeof window.onAuthChanged === 'function') {
        window.onAuthChanged(window.sbUser, window.sbSubscribed);
      }
    }
  }
}

// ── DOMContentLoaded ───────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  // Wire up modal UI regardless of SDK status
  document.getElementById('auth-modal')?.addEventListener('click', e => {
    if (e.target === document.getElementById('auth-modal')) closeModal();
  });
  document.querySelectorAll('.modal-tab').forEach(t =>
    t.addEventListener('click', () => _tab(t.dataset.tab)));
  document.getElementById('si-form')?.addEventListener('submit', _onSignIn);
  document.getElementById('su-form')?.addEventListener('submit', _onSignUp);

  // Initialize Supabase — bail gracefully if SDK failed to load
  if (!window.supabase?.createClient) {
    console.warn('Supabase SDK not available — auth features disabled.');
    return;
  }
  _sb = window.supabase.createClient(SUPABASE_URL, SUPABASE_KEY);
  await _initAuth();
});
