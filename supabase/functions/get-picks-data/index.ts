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
    'Access-Control-Allow-Methods': 'GET, OPTIONS',
    'Vary': 'Origin',
  }
}

const ALLOWED_FILES = new Set(['picks_today', 'picks_history'])

serve(async (req) => {
  const CORS = corsHeaders(req)
  if (req.method === 'OPTIONS') return new Response('ok', { headers: CORS })
  if (req.method !== 'GET')     return json(CORS, { error: 'Method not allowed' }, 405)

  try {
    const supabase = createClient(
      Deno.env.get('SUPABASE_URL')!,
      Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!,
    )

    const authHeader = req.headers.get('Authorization')
    if (!authHeader) {
      return json(CORS, { error: 'Unauthorized' }, 401)
    }

    const { data: { user }, error: authErr } = await supabase.auth.getUser(
      authHeader.replace('Bearer ', '')
    )
    if (authErr || !user) return json(CORS, { error: 'Unauthorized' }, 401)

    const { data: profile } = await supabase
      .from('profiles')
      .select('subscription_status')
      .eq('user_id', user.id)
      .single()

    const ACTIVE_STATUSES = new Set(['active', 'trialing', 'lifetime'])
    if (!ACTIVE_STATUSES.has(profile?.subscription_status ?? '')) {
      return json(CORS, { error: 'Subscription required' }, 402)
    }

    const url  = new URL(req.url)
    const file = url.searchParams.get('file') ?? 'picks_today'
    if (!ALLOWED_FILES.has(file)) return json(CORS, { error: 'Invalid file' }, 400)

    const { data, error: storageErr } = await supabase.storage
      .from('picks-data')
      .download(`${file}.json`)

    if (storageErr || !data) return json(CORS, { error: 'Data not available yet' }, 503)

    return new Response(await data.text(), {
      headers: { ...CORS, 'Content-Type': 'application/json' },
    })
  } catch (err) {
    console.error('get-picks-data error:', err)
    return json(CORS, { error: 'Internal error' }, 500)
  }
})

function json(cors: Record<string, string>, body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { ...cors, 'Content-Type': 'application/json' },
  })
}
