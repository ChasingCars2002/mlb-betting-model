import { serve } from 'https://deno.land/std@0.168.0/http/server.ts'
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

const ALLOWED_ORIGINS = new Set([
  'https://baseballbettingbot.com',
  'https://www.baseballbettingbot.com',
  'http://localhost:8000',
  'http://127.0.0.1:8000',
])

function corsHeaders(req: Request): Record<string, string> {
  const origin = req.headers.get('Origin') ?? ''
  const allow  = ALLOWED_ORIGINS.has(origin) ? origin : 'https://baseballbettingbot.com'
  return {
    'Access-Control-Allow-Origin': allow,
    'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
    'Access-Control-Allow-Methods': 'POST, OPTIONS',
    'Vary': 'Origin',
  }
}

// Promo code → subscription_status grant
const PROMO_CODES: Record<string, string> = {
  'FREE4LIFE': 'lifetime',
}

serve(async (req) => {
  const CORS = corsHeaders(req)
  if (req.method === 'OPTIONS') return new Response('ok', { headers: CORS })
  if (req.method !== 'POST')    return json(CORS, { error: 'Method not allowed' }, 405)

  try {
    const supabase = createClient(
      Deno.env.get('SUPABASE_URL')!,
      Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!,
    )

    const authHeader = req.headers.get('Authorization')
    if (!authHeader) return json(CORS, { error: 'Unauthorized' }, 401)

    const { data: { user }, error } = await supabase.auth.getUser(
      authHeader.replace('Bearer ', '')
    )
    if (error || !user) return json(CORS, { error: 'Unauthorized' }, 401)
    if (!user.email_confirmed_at) {
      return json(CORS, { error: 'Email not confirmed' }, 403)
    }

    let body: unknown = {}
    try { body = await req.json() } catch { /* empty body */ }
    const rawCode = (body as { code?: unknown })?.code
    const code = String(rawCode ?? '').toUpperCase().trim()
    // Length cap prevents an attacker stuffing huge strings into the
    // PROMO_CODES lookup or the database column.
    if (!code || code.length > 32) {
      return json(CORS, { error: 'Promo code required' }, 400)
    }

    const grantStatus = PROMO_CODES[code]
    if (!grantStatus) return json(CORS, { error: 'Invalid promo code' }, 400)

    const { data: profile } = await supabase
      .from('profiles')
      .select('subscription_status, promo_code')
      .eq('user_id', user.id)
      .single()

    if (profile?.subscription_status === 'active' ||
        profile?.subscription_status === 'trialing') {
      return json(CORS, { error: 'You already have an active subscription' }, 400)
    }
    if (profile?.promo_code) {
      return json(CORS, { error: 'A promo code has already been applied to this account' }, 400)
    }

    // Only update rows that still match the "no promo applied" precondition.
    // This closes the TOCTOU window between the SELECT above and the UPDATE.
    const { data: updated, error: updateErr } = await supabase
      .from('profiles')
      .update({ subscription_status: grantStatus, promo_code: code })
      .eq('user_id', user.id)
      .is('promo_code', null)
      .select('user_id')

    if (updateErr || !updated?.length) {
      return json(CORS, { error: 'A promo code has already been applied to this account' }, 400)
    }

    return json(CORS, { success: true, status: grantStatus })
  } catch (err) {
    console.error('redeem-promo error:', err)
    return json(CORS, { error: 'Could not redeem promo code' }, 500)
  }
})

function json(cors: Record<string, string>, body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { ...cors, 'Content-Type': 'application/json' },
  })
}
