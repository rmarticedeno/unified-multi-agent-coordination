# Upstream-derived A2A Hello World interoperability fixture

This is a minimal copy of the official `a2a-samples` Python Hello World agent,
used only for third-party interoperability testing. `UPSTREAM.json` pins the
source commit, SDK version, Git blob identities, license, and the comment-only
local adaptations. The application structure and executor remain separate from
this project's fixture agent, so the check crosses an upstream-derived A2A
implementation boundary without claiming an byte-identical vendored tree.
