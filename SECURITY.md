# Security Policy

## Supported Versions

We release patches for security vulnerabilities for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

If you discover a security vulnerability within this project, please send an email to the repository owner. All security vulnerabilities will be promptly addressed.

Please include the following information in your report:

- Type of vulnerability
- Full path of the affected source file(s)
- Location of the affected code (tag/branch/commit or direct URL)
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

### What to Expect

- You will receive an acknowledgment within 48 hours
- We will investigate and provide an estimated timeline for a fix
- We will keep you informed of the progress
- You will be credited in the security advisory (unless you prefer to remain anonymous)

## Security Best Practices for Deployment

### Camera Credentials

**CRITICAL**: Never commit camera credentials to version control.

- ✅ Use `camera_credentials.yaml` (gitignored)
- ✅ Store credentials in environment variables or secrets manager
- ❌ Never hardcode passwords in code
- ❌ Never commit config files with real credentials

### Network Security

**Recommended deployment practices:**

1. **Firewall Configuration**
   - Only expose necessary ports (default: 8000 for web UI)
   - Restrict camera RTSP access to local network
   - Use VPN for remote access instead of port forwarding

2. **Camera Network Isolation**
   - Place cameras on isolated VLAN if possible
   - Use strong camera passwords (20+ characters)
   - Disable unused camera features (UPnP, P2P cloud)

3. **Web Interface Security**
   - Run behind reverse proxy (nginx/caddy) with HTTPS
   - Implement authentication if exposing to internet
   - Use rate limiting to prevent abuse

### System Security

**Running in production:**

```bash
# Run as non-root user
sudo useradd -r -s /bin/false telescope

# Restrict file permissions
chmod 600 camera_credentials.yaml
chmod 600 config/config.yaml

# Use systemd hardening (see SERVICE_SETUP.md)
ProtectSystem=strict
PrivateTmp=true
NoNewPrivileges=true
```

### Data Privacy

**Important considerations:**

- Recorded clips may contain sensitive information (people, property)
- Store clips in encrypted storage if possible
- Implement retention policies (auto-delete old clips)
- Comply with local privacy laws (GDPR, CCPA, etc.)
- Post signage if monitoring areas where privacy is expected

### Model Security

**ML Model considerations:**

- Models are downloaded from trusted sources (official repositories)
- Verify checksums of downloaded models if available
- Be aware that ML models can have biases or limitations
- Test thoroughly before deploying in safety-critical scenarios

## Known Security Considerations

### Current Limitations

1. **No Built-in Authentication**
   - Web interface has no login system
   - Suitable for trusted networks only
   - Use reverse proxy with auth for public deployment

2. **Unencrypted RTSP**
   - Camera streams use unencrypted RTSP by default
   - Credentials sent in clear text over local network
   - Mitigate with network isolation

3. **No Input Validation on Config**
   - Malformed YAML config could cause crashes
   - Only use trusted config files
   - Validate config before deployment

### Planned Security Improvements

- [ ] Add web UI authentication (OAuth2, basic auth)
- [ ] HTTPS support for web interface
- [ ] Config validation schema
- [ ] Encrypted storage for sensitive clips
- [ ] Audit logging for all detection events

## Security Disclosure Policy

We follow responsible disclosure practices:

1. Report received → Acknowledged within 48h
2. Investigation → Timeline provided within 7 days
3. Fix developed → Patch released ASAP
4. Public disclosure → After users have time to update (typically 30 days)

## Security Resources

- [OWASP IoT Security](https://owasp.org/www-project-internet-of-things/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [Camera Security Best Practices](https://www.cisa.gov/uscert/ncas/tips/ST15-002)

## Questions?

If you have questions about this security policy, please open a GitHub Discussion in the Security category.
