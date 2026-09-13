# Share the live demo remotely

The demo can be shared through a temporary Cloudflare Quick Tunnel, while this computer and its Internet connection remain on. Each invitation expires after 24 hours by default. Anyone holding the full invitation can open the live 3D experiment and films; it is not tied to a particular identity. Keep it within the intended audience.

Run `python scripts/share_demo.py --invite-file /tmp/fruit-fly-invite.txt` in one terminal and `cloudflared tunnel --url http://127.0.0.1:8788` in another. Use an unused invitation filename. The file is created with owner-only permissions. Append its `/invite/...` path to the HTTPS hostname printed by cloudflared. Send that complete link to the viewer. Do not add the invitation or tunnel logs to Git.

The invitation sets a secure HTTP-only cookie and redirects to the experiment. The sharing server allows only the demo assets, six films, selected documentation, and live experiment API. It blocks repository browsing, directory listings, model checkpoints, raw data, and unauthenticated access. Its experiment runs in a separate process from the local demo, so remote controls do not alter the local run. Multiple remote viewers share the remote run; this is a small temporary demo, not a multi-user hosting platform. Network latency slows live simulation playback; the finished films play at their rendered speed.

Stop both terminal processes with Ctrl+C to end sharing. Starting again produces a new invitation and usually a new tunnel hostname. A temporary tunnel is not permanent hosting. The GitHub repository remains private.

Reference: [Cloudflare Quick Tunnels](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/trycloudflare/).
