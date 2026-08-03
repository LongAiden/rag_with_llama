# Optional migrations

Postgres runs the `.sql` files sitting **directly** in the mounted
`/docker-entrypoint-initdb.d` directory. Subdirectories are ignored, so nothing
in here is applied automatically — that is the point.

## 001_create_graph_tables.sql

Tables for the knowledge graph feature (`src/app/graph/`). That feature is not
wired into the app: its router is never mounted, and no live code path imports
it. Applying this would create empty `entities` / `relationships` tables that
nothing reads.

Apply it only if you mount `graph_routes.router` in `src/app/api/app.py`:

```bash
docker compose exec -T postgres psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
  < deploy/migrations/optional/001_create_graph_tables.sql
```
