-- Review-only: finds teams created by the real-database tests that were never
-- cleaned up (a test that fails half-way skips its purge_team() in `finally`).
--
-- READ ONLY. Contains no DELETE/UPDATE. Run it inside a read-only transaction if
-- you want that enforced by the server as well:
--     BEGIN READ ONLY;  <this file>;  ROLLBACK;
--
-- The names below are the literal Team(name=...) values the real-DB tests insert
-- (tests/test_teams.py, tests/test_generation.py, tests/test_subscription_cancel.py).
-- A real customer team is very unlikely to share one, but check the counts and
-- created_at before deleting anything -- e.g. a `members` count above the 1-2 the
-- tests add would mean a real team happens to match.

SELECT
    t.id,
    t.name,
    t.created_at,
    t.deleted_at,
    (SELECT count(*) FROM team_members         m WHERE m.team_id = t.id) AS members,
    (SELECT count(*) FROM generation_jobs      j WHERE j.team_id = t.id) AS generation_jobs,
    (SELECT count(*) FROM team_subscriptions   s WHERE s.team_id = t.id) AS subscriptions,
    (SELECT count(*) FROM billing_transactions b WHERE b.team_id = t.id) AS billing_transactions,
    (SELECT count(*) FROM team_invites         i WHERE i.team_id = t.id) AS invites,
    t.subscription_credits_remaining,
    t.topup_credits_balance
FROM teams t
WHERE t.name IN (
        'Access gate test',
        'Generations filter test',
        'Owner-only delete test',
        'Library source mine',
        'Library source other',
        'Race boundary team'
      )
   OR t.name LIKE 'Boundary test day\_%' ESCAPE '\'
ORDER BY t.created_at, t.name;
