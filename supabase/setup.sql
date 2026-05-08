-- ============================================================
-- Supabase setup — run this once in the SQL Editor
-- Project: mlb-betting-model
-- ============================================================

-- 1. profiles table -------------------------------------------
CREATE TABLE IF NOT EXISTS public.profiles (
  user_id            UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  stripe_customer_id TEXT,
  subscription_status TEXT NOT NULL DEFAULT 'inactive',
  subscription_end   TIMESTAMPTZ,
  created_at         TIMESTAMPTZ DEFAULT NOW()
);

-- 2. Row Level Security ---------------------------------------
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;

-- Users can only read their own row
CREATE POLICY "Users read own profile"
  ON public.profiles FOR SELECT
  USING (auth.uid() = user_id);

-- 3. Auto-create profile on signup ----------------------------
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER LANGUAGE plpgsql SECURITY DEFINER AS $$
BEGIN
  INSERT INTO public.profiles (user_id)
  VALUES (NEW.id)
  ON CONFLICT (user_id) DO NOTHING;
  RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- ============================================================
-- Migration: promo code support (run after initial setup)
-- ============================================================
ALTER TABLE public.profiles
  ADD COLUMN IF NOT EXISTS promo_code TEXT;

-- Valid subscription_status values:
--   inactive  — no subscription
--   trialing  — in 5-day free trial (payment method collected, not yet charged)
--   active    — paid and active
--   past_due  — payment failed
--   lifetime  — permanent free access via promo code Free4life

-- ============================================================
-- After running this SQL:
--
-- 1. Go to Storage → New bucket
--    Name: picks-data
--    Public: OFF (private)
--
-- 2. Go to Authentication → Settings
--    Enable "Confirm email" (recommended)
--    Set Site URL to your GitHub Pages domain
--
-- 3. Set Edge Function secrets (Dashboard → Edge Functions → Secrets):
--    STRIPE_SECRET_KEY      = sk_live_...
--    STRIPE_WEBHOOK_SECRET  = whsec_...
--    STRIPE_PRICE_ID        = price_...
--    SITE_URL               = https://yourdomain.com
-- ============================================================
