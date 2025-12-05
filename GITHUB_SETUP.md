# Pushing to GitHub

## Repository Info
- Remote: https://github.com/RCSchmelzle/spot_teleop.git
- Branch: main

## Authentication Setup

GitHub no longer accepts password authentication. You need to use either:
1. **Personal Access Token (PAT)** - recommended for HTTPS
2. **SSH keys** - recommended for SSH

### Option 1: Personal Access Token (HTTPS)

1. Create a token on GitHub:
   - Go to: https://github.com/settings/tokens
   - Click "Generate new token" → "Generate new token (classic)"
   - Give it a name like "spot_teleop"
   - Select scopes: `repo` (all sub-scopes)
   - Click "Generate token"
   - **COPY THE TOKEN NOW** (you won't see it again!)

2. Configure git to cache credentials:
   ```bash
   git config --global credential.helper store
   # Or for temporary cache (1 hour):
   # git config --global credential.helper 'cache --timeout=3600'
   ```

3. Push to GitHub:
   ```bash
   cd ~/Projects/teleoperation_spot
   git remote add origin https://github.com/RCSchmelzle/spot_teleop.git
   git branch -M main
   git push -u origin main
   ```

4. When prompted:
   - Username: `RCSchmelzle`
   - Password: **paste your Personal Access Token**

### Option 2: SSH Keys (Recommended)

1. Generate SSH key (if you don't have one):
   ```bash
   ssh-keygen -t ed25519 -C "your_email@example.com"
   # Press Enter for default location
   # Optionally set a passphrase
   ```

2. Add key to ssh-agent:
   ```bash
   eval "$(ssh-agent -s)"
   ssh-add ~/.ssh/id_ed25519
   ```

3. Copy public key:
   ```bash
   cat ~/.ssh/id_ed25519.pub
   ```

4. Add to GitHub:
   - Go to: https://github.com/settings/keys
   - Click "New SSH key"
   - Paste your public key
   - Click "Add SSH key"

5. Use SSH URL instead:
   ```bash
   cd ~/Projects/teleoperation_spot
   git remote add origin git@github.com:RCSchmelzle/spot_teleop.git
   git branch -M main
   git push -u origin main
   ```

## Quick Commands (after auth is set up)

```bash
cd ~/Projects/teleoperation_spot

# Add remote (if not already added)
git remote add origin https://github.com/RCSchmelzle/spot_teleop.git
# OR for SSH: git remote add origin git@github.com:RCSchmelzle/spot_teleop.git

# Rename branch to main
git branch -M main

# Push to GitHub
git push -u origin main

# For future pushes:
git push
```

## Troubleshooting

### If remote already exists:
```bash
git remote remove origin
# Then add it again
```

### If push is rejected (diverged histories):
```bash
# Only if you're sure you want to overwrite remote
git push -f origin main
```

### Check current remote:
```bash
git remote -v
```

### Large repository warning:
After cleaning, the repo should be ~100-200MB. First push may take a while.
GitHub has a 100MB single file limit and warns at 50MB.
All our files should be under these limits after cleanup.