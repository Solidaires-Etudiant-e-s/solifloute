# SoliFloute cloud inference (Modal)

This directory contains the **cloud** processing target for SoliFloute, for
**videos only**. When a user picks the *Cloud* target on a video, the Nuxt server
proxies the request to a serverless GPU endpoint hosted on [Modal](https://modal.com)
instead of running detection/blurring locally on the SoliFloute server or in the
browser.

> **Privacy:** Modal advertises a zero-data-retention policy, but media is still
> transmitted to and processed by an external provider. It should not be used
> for extremely private content. The web UI warns the user before this target is
> used.

## What it does

`app.py` deploys a FastAPI **Web Function** exposing a single route:

| Route              | Body                                      | Returns                      |
| ------------------ | ----------------------------------------- | ---------------------------- |
| `POST /process-video` | multipart `file` + `settings` JSON    | blurred video (`video/mp4`)  |

`processing.py` and `tracking.py` are faithful Python ports of the TypeScript
pipelines in `shared/utils/` so the cloud results stay consistent with the local
server and browser targets. `video.py` handles frame extraction, temporal face
tracking and re-encoding with `ffmpeg`.

## Local development

With a Modal account and `modal` installed:

```bash
pip install -r modal/requirements.txt
modal serve modal/app.py        # ephemeral URL for local testing
```

## Deployment

```bash
modal deploy modal/app.py
```

This prints the public URL of the web function, e.g.
`https://yourworkspace--solifloute-cloud-faceprocessor-process-video.modal.run`.

## Configuring the Nuxt server

Set these environment variables so the Nuxt server knows how to reach Modal:

- `MODAL_VIDEO_URL` — URL of the `process-video` route.
- `MODAL_KEY` / `MODAL_SECRET` — optional proxy-token auth credentials. If set,
  they are sent as the `Modal-Key` and `Modal-Secret` headers on every request.
  If unset, requests are sent without authentication (only use that for testing).

If `MODAL_VIDEO_URL` is unset, the Cloud target returns a clear 501 error from
the SoliFloute server.
