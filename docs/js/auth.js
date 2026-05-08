/* auth.js — Supabase Auth + Stripe subscription management */
'use strict';

const SUPABASE_URL = 'https://hmukgvrpuncxkzeuujam.supabase.co';
const SUPABASE_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImhtdWtndnJwdW5jeGt6ZXV1amFtIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzgxOTMwNDgsImV4cCI6MjA5Mzc2OTA0OH0.0SC2M6Fro3dOwWwf-7Qld-aN3RjCusNkKUWVEalLtEM';
const PICKS_FN    = `${SUPABASE_URL}/functions/v1/get-picks-data`;
const CHECKOUT_FN = `${SUPABASE_URL}/functions/v1/create-checkout-session`;
const CANCEL_FN   = `${SUPABASE_URL}/functions/v1/cancel-subscription`;
const REDEEM_FN   = `${SUPABASE_URL}/functions/v1/redeem-promo`;

if (!window.supabase?.createClient) {
  console.warn('Supabase SDK not available — auth features disabled.');
}
const sb = window.supabase?.createClient(SUPABASE_URL, SUPABASE_KEY) ?? null;

// ── Shared auth state (read by dashboard.js) ───────────────────────────────
window.sbUser       = null;
window.sbSession    = null;
window.sbSubscribed = false;
window.sbSubStatus  = 'inactive';  // 'inactive' | 'trialing' | 'active' | 'past_due' | 'lifetime'
window.sbSubEnd     = null;        // ISO date string of period/trial end

// Resolves once the initial auth + subscription check completes (used to
// prevent loadGatedData from running before sbSubscribed is known).
let _resolveAuthReady;
window.sbAuthReady = new Promise(resolve => { _resolveAuthReady = resolve; });

// ── Subscription check ─────────────────────────────────────────────────────
async function _checkSub() {
  if (!window.sbUser || !sb) {
    window.sbSubscribed = false;
    window.sbSubStatus  = 'inactive';
    window.sbSubEnd     = null;
    return;
  }
  const { data } = await sb
    .from('profiles')
    .select('subscription_status, subscription_end')
    .eq('user_id', window.sbUser.id)
    .single();
  const status = data?.subscription_status ?? 'inactive';
  window.sbSubStatus  = status;
  window.sbSubEnd     = data?.subscription_end ?? null;
  window.sbSubscribed = ['active', 'trialing', 'lifetime'].includes(status);
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

// ── Stripe checkout (5-day free trial) ────────────────────────────────────
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
      btns.forEach(b => { b.disabled = false; b.textContent = 'Start Free Trial'; });
    }
  } catch {
    alert('Network error. Please try again.');
    btns.forEach(b => { b.disabled = false; b.textContent = 'Start Free Trial'; });
  }
};

// ── Cancel subscription ────────────────────────────────────────────────────
window.cancelSubscription = async function () {
  const btn = document.getElementById('cancel-sub-btn');
  if (btn) { btn.disabled = true; btn.textContent = 'Cancelling…'; }
  try {
    const res = await fetch(CANCEL_FN, {
      method: 'POST',
      headers: { Authorization: `Bearer ${window.sbSession.access_token}` },
    });
    const data = await res.json();
    if (res.ok) {
      const until = data.access_until
        ? new Date(data.access_until).toLocaleDateString('en-US', {
            month: 'short', day: 'numeric', year: 'numeric',
          })
        : 'the end of your billing period';
      const contentEl = document.getElementById('manage-modal-body');
      if (contentEl) {
        contentEl.innerHTML = `
          <div class="modal-success">
            <div class="success-title">Subscription Cancelled</div>
            <div class="success-body">
              Your access continues until <strong>${until}</strong>.<br>
              You won't be charged again.
            </div>
          </div>`;
      }
      setTimeout(async () => {
        await _checkSub();
        _updateHeader();
      }, 1500);
    } else {
      if (btn) { btn.disabled = false; btn.textContent = 'Cancel Subscription'; }
      alert(data.error || 'Could not cancel. Please try again.');
    }
  } catch {
    if (btn) { btn.disabled = false; btn.textContent = 'Cancel Subscription'; }
    alert('Network error. Please try again.');
  }
};

// ── Redeem promo code ──────────────────────────────────────────────────────
window.redeemPromo = async function () {
  if (!window.sbUser) { openModal('signup'); return; }
  const input = document.getElementById('promo-code-input');
  const btn   = document.getElementById('promo-submit-btn');
  const err   = document.getElementById('promo-error');
  if (!input || !btn) return;

  const code = input.value.trim();
  if (!code) {
    if (err) { err.textContent = 'Please enter a promo code.'; err.style.display = 'block'; }
    return;
  }

  btn.disabled = true; btn.textContent = 'Applying…';
  if (err) err.style.display = 'none';

  try {
    const res = await fetch(REDEEM_FN, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${window.sbSession.access_token}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ code }),
    });
    const data = await res.json();
    if (res.ok) {
      await _checkSub();
      _updateHeader();
      _hidePromoForm();
      if (typeof window.onAuthChanged === 'function') {
        window.onAuthChanged(window.sbUser, window.sbSubscribed);
      }
    } else {
      if (err) { err.textContent = data.error || 'Invalid promo code.'; err.style.display = 'block'; }
      btn.disabled = false; btn.textContent = 'Apply Code';
    }
  } catch {
    if (err) { err.textContent = 'Network error. Please try again.'; err.style.display = 'block'; }
    btn.disabled = false; btn.textContent = 'Apply Code';
  }
};

// ── Manage subscription modal ──────────────────────────────────────────────
window.openManageModal = function () {
  const m = document.getElementById('manage-modal');
  if (!m) return;
  _renderManageModal();
  m.style.display = 'flex';
};

window.closeManageModal = function () {
  const m = document.getElementById('manage-modal');
  if (m) m.style.display = 'none';
};

function _renderManageModal() {
  const el = document.getElementById('manage-modal-body');
  if (!el) return;

  const status   = window.sbSubStatus;
  const subEnd   = window.sbSubEnd;
  const fmtDate  = (iso) => iso
    ? new Date(iso).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
    : null;

  let pillClass = 'status-inactive';
  let pillLabel = 'Inactive';
  let meta      = '';
  let showCancel = false;

  if (status === 'lifetime') {
    pillClass = 'status-lifetime'; pillLabel = 'Lifetime Access';
    meta = 'You have permanent free access via promo code.';
  } else if (status === 'trialing') {
    pillClass = 'status-trial'; pillLabel = 'Free Trial';
    meta = subEnd ? `Trial ends <strong>${fmtDate(subEnd)}</strong> — card will be charged then.` : 'Trial active.';
    showCancel = true;
  } else if (status === 'active') {
    pillClass = 'status-active'; pillLabel = 'Active';
    meta = subEnd ? `Renews <strong>${fmtDate(subEnd)}</strong>` : '';
    showCancel = true;
  } else if (status === 'past_due') {
    pillClass = 'status-pastdue'; pillLabel = 'Past Due';
    meta = 'Payment failed. Please update your payment method.';
  }

  el.innerHTML = `
    <div class="manage-status-row">
      <span class="status-pill ${pillClass}">${pillLabel}</span>
      <span class="manage-meta">${meta}</span>
    </div>
    ${showCancel ? `
      <div class="manage-cancel-section">
        <p class="manage-cancel-info">Cancelling ends billing at the current period end — your access remains until then.</p>
        <button class="cancel-btn" id="cancel-sub-btn" onclick="window.cancelSubscription()">Cancel Subscription</button>
      </div>` : ''}`;
}

// ── Promo code form toggle (in paywall) ────────────────────────────────────
window.showPromoForm = function () {
  const form = document.getElementById('promo-inline-form');
  const link = document.getElementById('promo-toggle-link');
  if (form) form.style.display = 'flex';
  if (link) link.style.display = 'none';
};

function _hidePromoForm() {
  const form = document.getElementById('promo-inline-form');
  if (form) form.style.display = 'none';
}

// ── Header auth bar ────────────────────────────────────────────────────────
function _updateHeader() {
  const loggedOut = document.getElementById('auth-logged-out');
  const loggedIn  = document.getElementById('auth-logged-in');
  const emailEl   = document.getElementById('auth-user-email');
  const manageBtn = document.getElementById('auth-manage-btn');
  if (!loggedOut || !loggedIn) return;
  if (window.sbUser) {
    loggedOut.style.display = 'none';
    loggedIn.style.display  = 'flex';
    if (emailEl) emailEl.textContent = window.sbUser.email;
    if (manageBtn) manageBtn.style.display = window.sbSubscribed ? 'inline-block' : 'none';
  } else {
    loggedOut.style.display = 'flex';
    loggedIn.style.display  = 'none';
  }
}

// ── Auth actions ───────────────────────────────────────────────────────────
window.sbSignOut = async function () { if (sb) await sb.auth.signOut(); };

// ── Modal ──────────────────────────────────────────────────────────────────
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
  document.querySelectorAll('.modal-panel').forEach(p =>
    p.style.display = (p.dataset.panel === name) ? 'flex' : 'none');
}

function _err(msg) {
  document.querySelectorAll('.modal-error').forEach(el => {
    el.textContent = msg;
    el.style.display = msg ? 'block' : 'none';
  });
}
function _clearErr() { _err(''); }

// ── Form submissions ───────────────────────────────────────────────────────
async function _onSignIn(e) {
  e.preventDefault();
  if (!sb) { _err('Auth not available.'); return; }
  _clearErr();
  const email = document.getElementById('si-email').value.trim();
  const pw    = document.getElementById('si-pw').value;
  const btn   = document.getElementById('si-btn');
  btn.disabled = true; btn.textContent = 'Signing in…';
  try {
    const { error } = await sb.auth.signInWithPassword({ email, password: pw });
    if (error) throw error;
    closeModal();
  } catch (err) {
    _err(err.message);
    btn.disabled = false; btn.textContent = 'Sign in';
  }
}

async function _onSignUp(e) {
  e.preventDefault();
  if (!sb) { _err('Auth not available.'); return; }
  _clearErr();
  const email = document.getElementById('su-email').value.trim();
  const pw    = document.getElementById('su-pw').value;
  const btn   = document.getElementById('su-btn');
  btn.disabled = true; btn.textContent = 'Creating account…';
  // After email confirmation, return here with a flag so we can auto-launch
  // Stripe checkout once the user lands back on the site.
  const redirectUrl = `${window.location.origin}${window.location.pathname}?checkout=pending`;
  try {
    const { data, error } = await sb.auth.signUp({
      email,
      password: pw,
      options: { emailRedirectTo: redirectUrl },
    });
    if (error) throw error;

    // If a session is returned (email confirmation disabled), the user is
    // logged in immediately — send them straight to Stripe checkout so we
    // collect a payment method before the free trial begins.
    if (data?.session) {
      window.sbSession = data.session;
      window.sbUser    = data.user ?? data.session.user ?? null;
      btn.textContent  = 'Redirecting to checkout…';
      closeModal();
      await window.startSubscription();
      return;
    }

    // Otherwise email confirmation is required; the redirect URL above
    // brings them back with ?checkout=pending which triggers checkout.
    const safeEmail = _escapeHtml(email);
    document.getElementById('signup-panel').innerHTML = `
      <div class="modal-success">
        <div style="font-size:2rem;margin-bottom:10px">📧</div>
        <div class="success-title">Check your email</div>
        <div class="success-body">
          A confirmation link was sent to <strong>${safeEmail}</strong>.<br>
          Click it to confirm your account — you'll then be taken to checkout
          to start your 5-day free trial.
        </div>
      </div>`;
  } catch (err) {
    _err(err.message);
    btn.disabled = false; btn.textContent = 'Create account';
  }
}

function _escapeHtml(str) {
  return String(str).replace(/[&<>"']/g, ch => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[ch]));
}

// ── Init ───────────────────────────────────────────────────────────────────
async function _init() {
  if (!sb) { _resolveAuthReady(); _updateHeader(); return; }
  const { data: { session } } = await sb.auth.getSession();
  window.sbSession = session;
  window.sbUser    = session?.user ?? null;
  if (window.sbUser) await _checkSub();
  _updateHeader();

  sb.auth.onAuthStateChange(async (event, session) => {
    window.sbSession    = session;
    window.sbUser       = session?.user ?? null;
    window.sbSubscribed = false;
    window.sbSubStatus  = 'inactive';
    window.sbSubEnd     = null;
    if (window.sbUser) await _checkSub();
    _updateHeader();
    // Unblock any loadGatedData() calls waiting on the initial auth state.
    _resolveAuthReady();
    if (typeof window.onAuthChanged === 'function') {
      window.onAuthChanged(window.sbUser, window.sbSubscribed);
    }

    // If the user just signed up (email confirmation flow) and they don't
    // have an active subscription yet, send them to Stripe checkout so we
    // collect a payment method before the trial begins.
    if (event === 'SIGNED_IN' && window.sbUser && !window.sbSubscribed) {
      const params = new URLSearchParams(window.location.search);
      if (params.get('checkout') === 'pending') {
        window.history.replaceState({}, '', window.location.pathname);
        window.startSubscription();
      }
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

  // Handle a return-from-email-confirmation when the session was already
  // restored by getSession() above (so onAuthStateChange may not fire SIGNED_IN).
  if (params.get('checkout') === 'pending' && window.sbUser && !window.sbSubscribed) {
    window.history.replaceState({}, '', window.location.pathname);
    window.startSubscription();
  }
}

document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('auth-modal')?.addEventListener('click', e => {
    if (e.target === document.getElementById('auth-modal')) closeModal();
  });
  document.getElementById('manage-modal')?.addEventListener('click', e => {
    if (e.target === document.getElementById('manage-modal')) closeManageModal();
  });
  document.querySelectorAll('.modal-tab').forEach(t =>
    t.addEventListener('click', () => _tab(t.dataset.tab)));
  document.getElementById('si-form')?.addEventListener('submit', _onSignIn);
  document.getElementById('su-form')?.addEventListener('submit', _onSignUp);
  _init();
});
