---
title: Breast Cancer Detection AI Backend
emoji: "🏥"
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Breast Cancer Detection AI Backend

Docker Space for the FastAPI inference backend.

## Required Space Variables

- `MONGO_URL`
- `MONGO_DB_NAME`
- `CORS_ORIGINS`

## Notes

- The container serves the API on port `7860`.
- Uploads and generated outputs live on the container filesystem and are not persistent on free hardware.
- Model weights must be present in the repository under `weights/` when the Space is built.
