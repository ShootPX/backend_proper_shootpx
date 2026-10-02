-- Pending plan switch (upgrade whose payment has not completed yet).
--
-- NOT applied automatically -- this repo has no migration runner. Run it by hand
-- against the database BEFORE deploying the code that reads these columns
-- (an unmigrated database makes every team_subscriptions query fail).
--
-- Additive and nullable: the previous code keeps working against the migrated
-- schema, so it is safe to run first and deploy after.
--
-- A pending switch exists exactly when the four columns are non-null.

BEGIN;

ALTER TABLE team_subscriptions
    ADD COLUMN pending_subscription_id           uuid REFERENCES subscription (id),
    ADD COLUMN pending_razorpay_subscription_id  text UNIQUE,
    ADD COLUMN pending_switch_id                 uuid,
    ADD COLUMN pending_switch_expires_at         timestamptz;

ALTER TABLE team_subscriptions
    ADD CONSTRAINT team_subscriptions_pending_switch_all_or_none CHECK (
        (pending_subscription_id IS NULL) = (pending_razorpay_subscription_id IS NULL)
        AND (pending_subscription_id IS NULL) = (pending_switch_id IS NULL)
        AND (pending_subscription_id IS NULL) = (pending_switch_expires_at IS NULL)
    );

-- The expiry sweep only ever looks at rows that have a pending switch.
CREATE INDEX ix_team_subscriptions_pending_switch_expiry
    ON team_subscriptions (pending_switch_expires_at)
    WHERE pending_switch_expires_at IS NOT NULL;

COMMIT;

-- ROLLBACK (only if you must undo; drops any in-flight pending switches):
--   BEGIN;
--   DROP INDEX IF EXISTS ix_team_subscriptions_pending_switch_expiry;
--   ALTER TABLE team_subscriptions
--       DROP CONSTRAINT IF EXISTS team_subscriptions_pending_switch_all_or_none,
--       DROP COLUMN IF EXISTS pending_switch_expires_at,
--       DROP COLUMN IF EXISTS pending_switch_id,
--       DROP COLUMN IF EXISTS pending_razorpay_subscription_id,
--       DROP COLUMN IF EXISTS pending_subscription_id;
--   COMMIT;
--
-- Before rolling back, cancel any unpaid replacement subscriptions in Razorpay:
--   SELECT team_id, pending_razorpay_subscription_id
--   FROM team_subscriptions WHERE pending_razorpay_subscription_id IS NOT NULL;
