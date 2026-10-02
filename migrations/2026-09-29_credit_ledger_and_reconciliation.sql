-- Credit ledger (phase 1: money-in only) + subscription reconciliation.
--
-- NOT applied automatically -- this repo has no migration runner. Run it by hand
-- against the database BEFORE deploying the code that uses it. Apply
-- 2026-09-29_pending_subscription_switch.sql first (this file does not depend on
-- it at the SQL level, but the deployed code needs both).
--
-- Additive: one nullable column, one index, one new table, one backfill. The
-- previous code keeps working against the migrated schema. One statement group,
-- one transaction -- it either fully applies or not at all.

BEGIN;

-- 1. When the hourly reconciliation job last compared a subscription to Razorpay.
ALTER TABLE team_subscriptions ADD COLUMN last_reconciled_at timestamptz;

-- The job only ever scans subscriptions that have a Razorpay id and are still live.
CREATE INDEX ix_team_subscriptions_reconcile
    ON team_subscriptions (last_reconciled_at NULLS FIRST, current_period_end)
    WHERE razorpay_subscription_id IS NOT NULL
      AND status IN ('active', 'pending', 'halted');

-- 2. Append-only record of credits that came in. NOT the balance (that stays on
--    teams); an audit trail plus the idempotency guard for grants.
CREATE TABLE credit_ledger (
    id                        uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    team_id                   uuid NOT NULL REFERENCES teams (id),
    pool                      text NOT NULL CHECK (pool IN ('subscription', 'topup')),
    entry_type                text NOT NULL CHECK (entry_type IN
                                  ('subscription_grant', 'switch_transfer', 'topup_purchase')),
    amount                    integer NOT NULL,           -- signed change to that pool
    balance_after             integer,                    -- NULL only for backfilled rows
    idempotency_key           text NOT NULL UNIQUE,
    source                    text NOT NULL CHECK (source IN
                                  ('webhook', 'reconcile', 'worker', 'backfill')),
    razorpay_subscription_id  text,
    razorpay_payment_id       text,
    switch_id                 uuid,
    metadata                  jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at                timestamptz NOT NULL DEFAULT now()
);

-- The billing-history endpoint pages newest-first per team.
CREATE INDEX ix_credit_ledger_team_created
    ON credit_ledger (team_id, created_at DESC, id DESC);

-- 3. Backfill past paid top-ups so the history is not empty for existing teams.
--    Same key shape the code uses ('payment:<razorpay payment id>'), so a webhook
--    redelivery after this migration is a no-op rather than a duplicate entry.
--    Held / mismatched payments granted no credits and are not entries.
--    Subscription grants made before this migration are not backfilled: nothing
--    recorded them.
INSERT INTO credit_ledger
    (team_id, pool, entry_type, amount, balance_after, idempotency_key, source,
     razorpay_payment_id, metadata, created_at)
SELECT bt.team_id,
       'topup',
       'topup_purchase',
       bt.credits_added,
       NULL,
       'payment:' || bt.razorpay_payment_id,
       'backfill',
       bt.razorpay_payment_id,
       jsonb_build_object('amount', bt.amount, 'backfilled', true),
       bt.created_at
FROM billing_transactions bt
WHERE bt.type = 'credit_pack'
  AND bt.status = 'completed'
  AND bt.credits_added > 0
ON CONFLICT (idempotency_key) DO NOTHING;

COMMIT;

-- ROLLBACK (drops the ledger and everything recorded in it):
--   BEGIN;
--   DROP TABLE IF EXISTS credit_ledger;
--   DROP INDEX IF EXISTS ix_team_subscriptions_reconcile;
--   ALTER TABLE team_subscriptions DROP COLUMN IF EXISTS last_reconciled_at;
--   COMMIT;
