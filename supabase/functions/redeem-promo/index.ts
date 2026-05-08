import { serve } from 'https://deno.land/std@0.168.0/http/server.ts'
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

const CORS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

// Promo code → subscription_status grant
const PROMO_CODES: Record<string, string> = {
  'FREE4LIFE': 'lifetime',
}

serve(async (req) => {
  if (req.method === 'OPTIONS') return new Response('ok', { headers: CORS })

  try {
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

    const body = await req.json()
    const code = String(body?.code ?? '').toUpperCase().trim()
    if (!code) return json({ error: 'Promo code required' }, 400)

    const grantStatus = PROMO_CODES[code]
    if (!grantStatus) return json({ error: 'Invalid promo code' }, 400)

    const { data: profile } = await supabase
      .from('profiles')
      .select('subscription_status, promo_code')
      .eq('user_id', user.id)
      .single()

    if (profile?.subscription_status === 'active') {
      return json({ error: 'You already have an active paid subscription' }, 400)
    }
    if (profile?.promo_code) {
      return json({ error: 'A promo code has already been applied to this account' }, 400)
    }

    await supabase.from('profiles').update({
      subscription_status: grantStatus,
      promo_code: code,
    }).eq('user_id', user.id)

    return json({ success: true, status: grantStatus })
  } catch (err) {
    return json({ error: err.message }, 500)
  }
})

function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { ...CORS, 'Content-Type': 'application/json' },
  })
}
