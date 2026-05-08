import { serve } from 'https://deno.land/std@0.168.0/http/server.ts'
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'
import Stripe from 'https://esm.sh/stripe@13.2.0?target=deno'

const CORS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

serve(async (req) => {
  if (req.method === 'OPTIONS') return new Response('ok', { headers: CORS })

  try {
    const stripeKey = Deno.env.get('STRIPE_SECRET_KEY')
    const priceId   = Deno.env.get('STRIPE_PRICE_ID')
    const missing   = [!stripeKey && 'STRIPE_SECRET_KEY', !priceId && 'STRIPE_PRICE_ID'].filter(Boolean)
    if (missing.length) {
      console.error('Missing env vars:', missing.join(', '))
      return json({ error: `Server misconfiguration: missing ${missing.join(', ')}` }, 500)
    }

    const supabase = createClient(
      Deno.env.get('SUPABASE_URL')!,
      Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!,
    )

    const authHeader = req.headers.get('Authorization')
    if (!authHeader) return json({ error: 'Unauthorized' }, 401)

    const { data: { user }, error } = await supabase.auth.getUser(
      authHeader.replace('Bearer ', '')
    )
    if (error || !user) return json({ error: 'Unauthorized' }, 401)

    const stripe  = new Stripe(stripeKey!, {
      apiVersion: '2023-08-16',
      httpClient: Stripe.createFetchHttpClient(),
    })
    const siteUrl = Deno.env.get('SITE_URL') ?? 'https://localhost'

    const session = await stripe.checkout.sessions.create({
      payment_method_types: ['card'],
      line_items: [{ price: priceId!, quantity: 1 }],
      mode: 'subscription',
      success_url: `${siteUrl}?subscribed=true`,
      cancel_url:  `${siteUrl}`,
      customer_email: user.email,
      metadata: { user_id: user.id },
    })

    return json({ url: session.url })
  } catch (err) {
    console.error('Checkout error:', err)
    return json({ error: err.message }, 500)
  }
})

function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { ...CORS, 'Content-Type': 'application/json' },
  })
}
