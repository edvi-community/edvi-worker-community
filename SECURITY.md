# Security

Use the repository’s **Security → Report a vulnerability** form to report
vulnerabilities privately. Keep credentials and exploit details out of public
issues and pull requests.

The coordinator binds to loopback and uses one shared token. Anyone with the
token can submit jobs and read job history. Use a trusted SSH tunnel for
development from another machine. Pick trusted model repositories and protect
local state, including the SQLite database.
