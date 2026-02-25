# Release Setup

This project uses GitHub Actions for automated releases via semantic-release.

## How It Works

1. Push to `main` with a conventional commit message triggers the Release workflow
2. semantic-release determines version bump based on commit type:
   - `feat:` → minor bump (e.g., 1.5.0 → 1.6.0)
   - `fix:` → patch bump (e.g., 1.5.0 → 1.5.1)
   - `feat!:` or `fix!:` → major bump (e.g., 1.5.0 → 2.0.0)
3. GitHub Actions bot creates the release, tag, and updates version files
4. The Release workflow also updates the webpage version and triggers GitHub Pages deployment

## Commit Message Convention

Use https://www.conventionalcommits.org/[Conventional Commits]:

```
feat: add new feature
fix: resolve bug
docs: update documentation
chore: maintenance (no release)
```

## Workflows

/release.yml` - Handles version- `.github/workflows bumps and releases
- `.github/workflows/pages.yml` - Deploys documentation site to GitHub Pages

## Local Testing

To test releases locally:
```bash
pip install python-semantic-release
semantic-release version --dry-run
```
