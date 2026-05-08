import { serve } from 'https://deno.land/std@0.168.0/http/server.ts'
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

const CORS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

const ALLOWED_FILES = new Set(['picks_today', 'picks_history'])

serve(async (req) => {
  if (req.method === 'OPTIONS') return new Response('ok', { headers: CORS })

  try {
    const supabase = createClient(
      Deno.env.get('SUPABASE_URL')!,
      Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!,
    )

    const authHeader = req.headers.get('Authorization')
    if (!authHeader) {
      return json({ error: 'Unauthorized' }, 401)
    }

    const { data: { user }, error: authErr } = await supabase.auth.getUser(
      authHeader.replace('Bearer ', '')
    )
    if (authErr || !user) return json({ error: 'Unauthorized' }, 401)

    const { data: profile } = await supabase
      .from('profiles')
      .select('subscription_status')
      .eq('user_id', user.id)
      .single()

    const ACTIVE_STATUSES = new Set(['active', 'trialing', 'lifetime'])
    if (!ACTIVE_STATUSES.has(profile?.subscription_status ?? '')) {
      return json({ error: 'Subscription required' }, 402)
    }

    const url  = new URL(req.url)
    const file = url.searchParams.get('file') ?? 'picks_today'
    if (!ALLOWED_FILES.has(file)) return json({ error: 'Invalid file' }, 400)

    const { data, error: storageErr } = await supabase.storage
      .from('picks-data')
      .download(`${file}.json`)

    if (storageErr || !data) return json({ error: 'Data not available yet' }, 503)

    return new Response(await data.text(), {
      headers: { ...CORS, 'Content-Type': 'application/json' },
    })
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
