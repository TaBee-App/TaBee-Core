# TaBee Frontend

React product shell for viewing TaBee's generated tablature.

## Development

```bash
npm install
npm run dev
```

The Vite dev server proxies:

- `/api` to `http://127.0.0.1:8000`
- `/uploads` to `http://127.0.0.1:8000`

That matches the current FastAPI prototype and can later point to the Spring Boot backend without changing the UI components.

## First Milestone

- Upload an audio file.
- Send it to `/api/generate-tab`.
- Store the returned AlphaTex locally.
- Render the tab through AlphaTab.
- Control playback speed, loop, and auto-scroll.
