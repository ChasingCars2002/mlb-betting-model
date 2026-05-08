import { serve } from 'https://deno.land/std@0.168.0/http/server.ts'
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'
import Stripe from 'https://esm.sh/stripe@13.2.0?target=deno'

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

    const { data: profile } = await supabase
      .from('profiles')
      .select('stripe_customer_id, subscription_status')
      .eq('user_id', user.id)
      .single()

    if (!profile?.stripe_customer_id) {
      return json(CORS, { error: 'No active subscription found' }, 400)
    }

    if (profile.subscription_status === 'lifetime') {
      return json(CORS, { error: 'Lifetime access cannot be cancelled' }, 400)
    }

    const stripe = new Stripe(Deno.env.get('STRIPE_SECRET_KEY')!, {
      apiVersion: '2023-08-16',
      httpClient: Stripe.createFetchHttpClient(),
    })

    // Find active or trialing subscription
    let sub: Stripe.Subscription | null = null
    for (const status of ['active', 'trialing'] as const) {
      const list = await stripe.subscriptions.list({
        customer: profile.stripe_customer_id,
        status,
        limit: 1,
      })
      if (list.data.length > 0) { sub = list.data[0]; break }
    }

    if (!sub) return json(CORS, { error: 'No active subscription found' }, 400)

    const updated = await stripe.subscriptions.update(sub.id, {
      cancel_at_period_end: true,
    })

    return json(CORS, {
      cancelled: true,
      access_until: new Date(updated.current_period_end * 1000).toISOString(),
    })
  } catch (err) {
    console.error('cancel-subscription error:', err)
    return json(CORS, { error: 'Could not cancel subscription' }, 500)
  }
})

function json(cors: Record<string, string>, body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { ...cors, 'Content-Type': 'application/json' },
  })
}
