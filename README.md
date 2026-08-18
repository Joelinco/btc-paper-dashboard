# BTC Paper Trading Dashboard

Upload all files in this folder to a new GitHub repository. The included GitHub Action runs daily at 00:15 UTC, updates the paper account, and publishes the `docs` folder through GitHub Pages.

This is a forward-only virtual account. It cannot access an exchange or place real orders.

## GitHub setup

1. Create a new public repository named `btc-paper-dashboard`.
2. Upload every file and folder from this package, including `.github`.
3. Open **Settings → Pages** and choose **GitHub Actions** under Source.
4. Open **Actions → Update BTC paper dashboard → Run workflow**.
5. When the action finishes, the Pages address appears in **Settings → Pages**.

