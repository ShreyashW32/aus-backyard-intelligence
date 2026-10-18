# Security

## Reporting a vulnerability

Please report security issues privately through GitHub's
[private vulnerability reporting](../../security/advisories/new) rather than a
public issue. I aim to respond within a week.

## Secrets

- API keys are read only from environment variables (`ABI_GOOGLE_MAPS_API_KEY`).
  Nothing secret is stored in the repository.
- Keep local keys in `.env` (git-ignored); `.env.example` lists the variables.
- Use a server-side Google key restricted to the Geocoding API and your server's
  IP address. Never expose it in browser code.

## Deploying publicly

- Set `ABI_DEV_API=0` to disable the dev/eval endpoints.
- Google geocoding results are held only in memory for the scan session and are
  never persisted.
- Raw property datasets, user feedback and scan output are excluded from git to
  protect privacy.
