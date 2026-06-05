Pin all dependencies to specific versions - example: `requests==2.31.0` instead of `requests>=2.0.0`
```
ignore-scripts true
allow-git none
```

Require a package cooldown period on builds, ref `pnpm` (move to PNPM) security docs
```
minimumReleaseAge: 20160  # 2 weeks (in minutes)
```

Disable `pre-install` and `post-install` scripts in package manager
```
onlyBuiltDependencies:
ignoredBuiltDependencies:
```

Use `npq` for hardening package installs
```
NPQ_PKG_MGR=pnpm npq install fastify
```

Use Socket Firewall (sfw) for blocking malicious packages
```
npm install -g sfw
sfw pnpm add express
```

Prevent npm lockfile injection
Enable 2FA for npm accounts

On vscode/Pycharm
Audit currently installed extensions
Remove unused extensions
