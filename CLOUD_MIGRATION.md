# Project X cloud operation

## Architecture

- Render web service: Streamlit display and lightweight per-symbol reads only.
- GitHub Actions: weekday batch collection for the all-TSE watchlist and JP/US day-trade watchlists.
- PostgreSQL: holdings, trade journal, settings, prediction/ranking history, candidate observations, current watchlists, and job status.
- GitHub: source of truth for code and versioned models. Passwords and database URLs are secrets, never files.

The 2.6 GB local market SQLite database and the 1.6 GB training CSV are not copied into the web service. They belong in an offline/object-storage research pipeline, because loading them into a 512 MB free web instance would make the app unstable.

## Safe migration

1. Create a PostgreSQL database and copy its connection URL.
2. On the current PC, set `DATABASE_URL` only for the current shell.
3. Run `python state_admin.py migrate`. This creates an ignored ZIP backup before uploading local mutable state.
4. Add the same `DATABASE_URL` and `PROJECT_X_ENV=production` to Render secrets.
5. Add `DATABASE_URL` to the GitHub repository Actions secrets.
6. Redeploy Render, run the workflow manually once, then check `python state_admin.py status` or the Actions log.

Do not delete the local backup until holdings, trade records, settings, watchlists, and candidate history are verified after a Render restart.

## Recovery

Download the most recent database export before schema or model changes. The local migration ZIP can be inspected without the application. To restore an individual item, extract it locally and run `state_admin.py migrate` with the intended environment namespace. Use `PROJECT_X_ENV=development` for tests and `production` only for verified releases.

## Deployment flow

Create changes on a branch, let `Project X tests` pass, merge to `main`, and allow Render auto-deploy. Production data remains in PostgreSQL and is not replaced by a deploy. Use a separate `PROJECT_X_ENV=development` namespace for preview/testing.
