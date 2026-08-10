import type { Face } from '~~/shared/types/faces'

export interface StoredEntry {
  id: string
  createdAt: number
  fileName: string
  mediaKind: 'image' | 'video'
  faces: Face[]
  status: string
  error: string
  warning: string
  lastDurationMs: number | null
  serverJobId: string | null
  originalBlob: Blob
  processedBlob: Blob | null
}

const DB_NAME = 'solifloute-entries'
const DB_VERSION = 1
const STORE_NAME = 'entries'

let dbPromise: Promise<IDBDatabase> | null = null

function openDb() {
  if (!dbPromise) {
    dbPromise = new Promise((resolve, reject) => {
      const request = indexedDB.open(DB_NAME, DB_VERSION)

      request.onupgradeneeded = () => {
        const db = request.result

        if (!db.objectStoreNames.contains(STORE_NAME)) {
          db.createObjectStore(STORE_NAME, { keyPath: 'id' })
        }
      }

      request.onsuccess = () => {
        resolve(request.result)
      }

      request.onerror = () => {
        dbPromise = null
        reject(request.error)
      }
    })
  }

  return dbPromise
}

function runTransaction<T>(
  mode: IDBTransactionMode,
  operation: (store: IDBObjectStore) => IDBRequest<T>
): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    void openDb().then((db) => {
      const transaction = db.transaction(STORE_NAME, mode)
      const store = transaction.objectStore(STORE_NAME)
      let result: T

      transaction.oncomplete = () => {
        resolve(result)
      }

      transaction.onerror = () => {
        reject(transaction.error)
      }

      transaction.onabort = () => {
        reject(transaction.error)
      }

      const request = operation(store)

      request.onsuccess = () => {
        result = request.result
      }

      request.onerror = () => {
        reject(request.error)
      }
    }).catch(reject)
  })
}

export function saveEntry(entry: StoredEntry) {
  return runTransaction('readwrite', store => store.put(entry)).then(() => undefined)
}

export function deleteEntry(id: string) {
  return runTransaction('readwrite', store => store.delete(id)).then(() => undefined)
}

export async function loadAllEntries(): Promise<StoredEntry[]> {
  return await runTransaction('readonly', store => store.getAll())
}
