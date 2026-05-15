# TaBee-Core
Core logic part of the project TaBee

## Backend Secrets

Do not commit real database passwords. Put local values in `backend/.env`; Git ignores
that file. Use `backend/.env.example` as the template.

```powershell
cd backend
Copy-Item .env.example .env
notepad .env
.\run-local.ps1
```

The backend can generate tabs by calling the local Python DSP pipeline:

```text
POST /api/audio-files/{audioFileId}/process
```

This creates a `tabs` row, a `tab_data` row, and `note_events` rows. The response
contains the generated `tabId`.
