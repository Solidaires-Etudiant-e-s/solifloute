import { mkdirSync } from 'node:fs'
import { createRequire } from 'node:module'
import { join } from 'node:path'

const JOB_STORE_ROOT = process.env.PROCESS_JOB_STORE || join(process.cwd(), '.data', 'video-jobs')
const JOB_DB_PATH = join(JOB_STORE_ROOT, 'jobs.sqlite')

export interface JobRow {
  id: string
  owner_id: string
  kind: string
  file_name: string
  mime_type: string
  status: string
  progress: number
  error: string
  created_at: number
  updated_at: number
  duration_ms: number | null
  output_path: string
  temp_root: string
  input_path: string
  settings_json: string
}

interface SqliteStatement {
  run(...params: unknown[]): unknown
  get(...params: unknown[]): JobRow | undefined
  all(...params: unknown[]): JobRow[]
}

interface SqliteDriver {
  exec(sql: string): void
  prepare(sql: string): SqliteStatement
  close(): void
}

function isBunRuntime() {
  return typeof process !== 'undefined' && typeof process.versions?.bun === 'string'
}

let db: SqliteDriver | null = null

function openDriver(): SqliteDriver {
  mkdirSync(JOB_STORE_ROOT, { recursive: true })
  const nodeRequire = createRequire(import.meta.url)

  if (isBunRuntime()) {
    const { Database } = nodeRequire('bun:sqlite') as {
      Database: new (path: string, options?: { create?: boolean }) => SqliteDriver
    }
    return new Database(JOB_DB_PATH, { create: true })
  }

  try {
    const { DatabaseSync } = nodeRequire('node:sqlite') as {
      DatabaseSync: new (path: string) => SqliteDriver
    }
    return new DatabaseSync(JOB_DB_PATH)
  } catch (cause) {
    throw new Error('Le stockage des taches exige Node.js >= 22.13 (node:sqlite) ou Bun.', { cause })
  }
}

export function getJobDb() {
  if (!db) {
    db = openDriver()
    db.exec(`
      PRAGMA journal_mode = WAL;
      PRAGMA synchronous = NORMAL;
      CREATE TABLE IF NOT EXISTS jobs (
        id TEXT PRIMARY KEY,
        owner_id TEXT NOT NULL,
        kind TEXT NOT NULL DEFAULT 'video-process',
        file_name TEXT NOT NULL,
        mime_type TEXT NOT NULL DEFAULT 'video/mp4',
        status TEXT NOT NULL,
        progress REAL NOT NULL DEFAULT 0,
        error TEXT NOT NULL DEFAULT '',
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL,
        duration_ms INTEGER,
        output_path TEXT NOT NULL DEFAULT '',
        temp_root TEXT NOT NULL DEFAULT '',
        input_path TEXT NOT NULL DEFAULT '',
        settings_json TEXT NOT NULL DEFAULT '{}'
      );
      CREATE INDEX IF NOT EXISTS idx_jobs_owner_created ON jobs(owner_id, created_at DESC);
    `)
  }

  return db
}

export function upsertJobRow(row: JobRow) {
  getJobDb().prepare(`
    INSERT INTO jobs (
      id, owner_id, kind, file_name, mime_type, status, progress, error,
      created_at, updated_at, duration_ms, output_path, temp_root, input_path, settings_json
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(id) DO UPDATE SET
      owner_id = excluded.owner_id,
      kind = excluded.kind,
      file_name = excluded.file_name,
      mime_type = excluded.mime_type,
      status = excluded.status,
      progress = excluded.progress,
      error = excluded.error,
      updated_at = excluded.updated_at,
      duration_ms = excluded.duration_ms,
      output_path = excluded.output_path,
      temp_root = excluded.temp_root,
      input_path = excluded.input_path,
      settings_json = excluded.settings_json
  `).run(
    row.id,
    row.owner_id,
    row.kind,
    row.file_name,
    row.mime_type,
    row.status,
    row.progress,
    row.error,
    row.created_at,
    row.updated_at,
    row.duration_ms,
    row.output_path,
    row.temp_root,
    row.input_path,
    row.settings_json
  )
}

export function allJobRows(): JobRow[] {
  return getJobDb().prepare(`
    SELECT * FROM jobs ORDER BY created_at ASC
  `).all()
}

export function deleteJobRow(id: string) {
  getJobDb().prepare(`DELETE FROM jobs WHERE id = ?`).run(id)
}
