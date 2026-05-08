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

const TRIAL_DAYS = 5

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

    const stripe  = new Stripe(Deno.env.get('STRIPE_SECRET_KEY')!, {
      apiVersion: '2023-08-16',
      httpClient: Stripe.createFetchHttpClient(),
    })
    const siteUrl = Deno.env.get('SITE_URL') ?? 'https://localhost'
    const priceId = Deno.env.get('STRIPE_PRICE_ID')!

    const session = await stripe.checkout.sessions.create({
      payment_method_types: ['card'],
      line_items: [{ price: priceId, quantity: 1 }],
      mode: 'subscription',
      allow_promotion_codes: true,
      // Always require a card up front so we can charge them when the
      // trial ends.
      payment_method_collection: 'always',
      subscription_data: {
        trial_period_days: TRIAL_DAYS,
        // If the card disappears before the trial ends, cancel rather
        // than letting the user keep access without a payment method.
        trial_settings: {
          end_behavior: { missing_payment_method: 'cancel' },
        },
      },
      success_url: `${siteUrl}?subscribed=true`,
      cancel_url:  `${siteUrl}`,
      customer_email: user.email,
      metadata: { user_id: user.id },
    })

    return json(CORS, { url: session.url })
  } catch (err) {
    console.error('create-checkout-session error:', err)
    return json(CORS, { error: 'Could not start checkout' }, 500)
  }
})

function json(cors: Record<string, string>, body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { ...cors, 'Content-Type': 'application/json' },
  })
}
