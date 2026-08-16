import { randomUUID } from 'node:crypto'
import { getCookie, setCookie, type H3Event } from 'h3'

const CLIENT_COOKIE_NAME = 'solitools_client_id'
const CLIENT_COOKIE_MAX_AGE_SECONDS = 24 * 60 * 60

function isUuid(value: string) {
  return /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(value)
}

export function getOrCreateClientId(event: H3Event) {
  const existingValue = getCookie(event, CLIENT_COOKIE_NAME)

  if (existingValue && isUuid(existingValue)) {
    return existingValue
  }

  const nextValue = randomUUID()

  setCookie(event, CLIENT_COOKIE_NAME, nextValue, {
    path: '/',
    maxAge: CLIENT_COOKIE_MAX_AGE_SECONDS,
    sameSite: 'lax',
    httpOnly: true
  })

  return nextValue
}
