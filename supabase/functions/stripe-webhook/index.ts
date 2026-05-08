import { serve } from 'https://deno.land/std@0.168.0/http/server.ts'
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'
import Stripe from 'https://esm.sh/stripe@13.2.0?target=deno'

serve(async (req) => {
  if (req.method !== 'POST') {
    return new Response('Method not allowed', { status: 405 })
  }
  const sig = req.headers.get('stripe-signature')
  if (!sig) return new Response('Missing signature', { status: 400 })
  const body = await req.text()

  const stripe = new Stripe(Deno.env.get('STRIPE_SECRET_KEY')!, {
    apiVersion: '2023-08-16',
    httpClient: Stripe.createFetchHttpClient(),
  })

  let event: Stripe.Event
  try {
    event = await stripe.webhooks.constructEventAsync(
      body, sig, Deno.env.get('STRIPE_WEBHOOK_SECRET')!
    )
  } catch (err) {
    console.error('stripe-webhook signature verification failed:', err)
    return new Response('Invalid signature', { status: 400 })
  }

  const supabase = createClient(
    Deno.env.get('SUPABASE_URL')!,
    Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!,
  )

  // Never overwrite a lifetime promo grant with a Stripe event
  const setStatus = async (customerId: string, status: string, periodEnd?: number) => {
    const { data } = await supabase
      .from('profiles')
      .select('user_id, subscription_status')
      .eq('stripe_customer_id', customerId)
      .single()
    if (data && data.subscription_status !== 'lifetime') {
      await supabase.from('profiles').update({
        subscription_status: status,
        subscription_end: periodEnd ? new Date(periodEnd * 1000).toISOString() : null,
      }).eq('user_id', data.user_id)
    }
  }

  switch (event.type) {
    case 'checkout.session.completed': {
      const s = event.data.object as Stripe.Checkout.Session
      if (s.customer && s.metadata?.user_id && s.subscription) {
        // Retrieve the subscription to check if it's in trial
        const sub = await stripe.subscriptions.retrieve(s.subscription as string)
        const status = sub.status === 'trialing' ? 'trialing' : 'active'
        const { data: profile } = await supabase
          .from('profiles')
          .select('subscription_status')
          .eq('user_id', s.metadata.user_id)
          .single()
        if (profile?.subscription_status !== 'lifetime') {
          await supabase.from('profiles').update({
            stripe_customer_id: s.customer as string,
            subscription_status: status,
            subscription_end: sub.current_period_end
              ? new Date(sub.current_period_end * 1000).toISOString()
              : null,
          }).eq('user_id', s.metadata.user_id)
        }
      }
      break
    }
    case 'customer.subscription.updated': {
      const sub = event.data.object as Stripe.Subscription
      let status: string
      if (sub.status === 'trialing') {
        status = 'trialing'
      } else if (sub.status === 'active') {
        status = 'active'
      } else if (sub.status === 'past_due') {
        status = 'past_due'
      } else {
        status = 'inactive'
      }
      await setStatus(sub.customer as string, status, sub.current_period_end)
      break
    }
    case 'customer.subscription.deleted': {
      const sub = event.data.object as Stripe.Subscription
      await setStatus(sub.customer as string, 'inactive')
      break
    }
    case 'invoice.payment_failed': {
      const inv = event.data.object as Stripe.Invoice
      await setStatus(inv.customer as string, 'past_due')
      break
    }
  }

  return new Response(JSON.stringify({ received: true }), {
    headers: { 'Content-Type': 'application/json' },
  })
})
