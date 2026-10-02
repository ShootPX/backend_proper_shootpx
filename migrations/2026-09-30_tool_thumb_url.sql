-- Tool card thumbnail URL, served by GET /landing/tools as `thumbUrl`.
--
-- NOT applied automatically -- this repo has no migration runner. Run it by hand
-- against the database BEFORE deploying the code that reads this column
-- (an unmigrated database makes every tool_definitions query fail).
--
-- Additive and nullable: the previous code keeps working against the migrated
-- schema, so it is safe to run first and deploy after.

BEGIN;

ALTER TABLE tool_definitions
    ADD COLUMN thumb_url text;

COMMIT;

-- Set thumbnails once the images are uploaded to public storage, e.g.:
--   UPDATE tool_definitions
--   SET thumb_url = 'https://<project>.supabase.co/storage/v1/object/public/site-assets/tool-thumbs/recolor-v1.webp'
--   WHERE feature_type = 'recolor';
--
-- /landing/tools is cached in Redis (1h), so clear it after updating rows:
--   POST /admin/cache/clear  (header x-cache-secret)
--
-- ROLLBACK:
--   ALTER TABLE tool_definitions DROP COLUMN IF EXISTS thumb_url;
