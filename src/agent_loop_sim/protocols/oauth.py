"""MCP authorisation as a state machine: the OAuth 2.1 flow the 2026-07-28 revision requires of
an HTTP MCP server and its client, and deliberately mis-configured variants that fail at the
step the spec says they must.

The flow (MCP spec 2026-07-28, "Authorization"): an unauthenticated request gets ``401`` with
``WWW-Authenticate: Bearer resource_metadata=…``; the client reads the protected-resource
metadata (RFC 9728), then the authorisation server's metadata (RFC 8414), checks the issuer
and that PKCE with S256 is supported, registers (a Client ID Metadata Document by default;
Dynamic Client Registration, now deprecated, as a variant), sends the user to the
authorisation endpoint with a PKCE challenge and the ``resource`` parameter (RFC 8707),
checks ``state`` and ``iss`` (RFC 9207) on the way back, redeems the code with the verifier,
and finally calls the MCP server with a bearer token whose audience is that server.

Tokens here are opaque strings with their claims kept beside them (a real server would verify
a signed JWT or introspect); the random values come from the seeded generator so a run is
reproducible in Python and TypeScript. PKCE is real: challenge = BASE64URL(SHA-256(verifier)).
"""
from __future__ import annotations

import base64
import hashlib
from typing import Any

from ..jsonfmt import compact
from ..rng import Rng

MCP = "https://mcp.example.com"
RESOURCE = MCP + "/mcp"
AS = "https://auth.example.com"
OTHER_RESOURCE = "https://calendar.example.com/mcp"
EVIL_AS = "https://evil.example.com"
CLIENT_ID_URL = "https://client.example.com/oauth/client.json"
REDIRECT = "http://127.0.0.1:33418/callback"
UPSTREAM = "https://api.files.example.com"
UNRESERVED = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~"

VARIANTS: dict[str, str] = {
    "ok": "Everything configured correctly (Client ID Metadata Document registration)",
    "dcr": "Dynamic Client Registration instead (deprecated in 2026-07-28, still allowed)",
    "no_pkce_support": "The authorisation server's metadata does not advertise PKCE",
    "pkce_skipped": "The client leaves out the PKCE challenge",
    "wrong_verifier": "The token request carries the wrong code verifier",
    "issuer_mismatch": "The metadata names a different issuer from the one asked",
    "iss_mismatch": "The authorisation response comes back with a different iss (mix-up)",
    "wrong_audience": "The client presents a token minted for another server",
    "insufficient_scope": "The token's scope is too narrow for the tool being called",
    "token_passthrough": "The MCP server forwards the client's token to an upstream API",
}


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def pkce_challenge(verifier: str) -> str:
    """RFC 7636 S256: BASE64URL-ENCODE(SHA256(ASCII(code_verifier)))."""
    return b64url(hashlib.sha256(verifier.encode("utf-8")).digest())


def random_string(rng: Rng, n: int) -> str:
    out = ""
    for _ in range(n):
        out += UNRESERVED[rng.randint(0, len(UNRESERVED) - 1)]
    return out


def pct(s: str) -> str:
    """Percent-encode everything but the RFC 3986 unreserved characters (UTF-8, upper-case hex)."""
    out = ""
    for b in s.encode("utf-8"):
        c = chr(b)
        if c in UNRESERVED:
            out += c
        else:
            out += "%" + format(b, "02X")
    return out


def form(pairs: list[list[str]]) -> str:
    return "&".join(pct(k) + "=" + pct(v) for k, v in pairs)


def run_oauth(variant: str = "ok", seed: int = 7) -> dict[str, Any]:
    """Play the flow. Returns ``steps`` (each an HTTP message, a check or a user action, with
    who sends it to whom) and ``outcome`` (``ok``, and if not, the step and the reason)."""
    if variant not in VARIANTS:
        raise ValueError("unknown variant " + variant)
    rng = Rng(seed)
    steps: list[dict[str, Any]] = []
    state: dict[str, Any] = {"phase": "unauthenticated"}

    def http(frm: str, to: str, label: str, method: str | None, url: str | None, headers: list[list[str]],
             body: Any, status: int | None, note: str = "") -> dict[str, Any]:
        s = {"seq": len(steps), "kind": "http", "from": frm, "to": to, "label": label, "method": method, "url": url,
             "headers": headers, "body": body, "status": status, "note": note, "ok": status is None or status < 400,
             "phase": state["phase"]}
        steps.append(s)
        return s

    def check(actor: str, label: str, ok: bool, detail: str, rule: str = "") -> bool:
        steps.append({"seq": len(steps), "kind": "check", "from": actor, "to": actor, "label": label, "ok": ok,
                      "detail": detail, "rule": rule, "phase": state["phase"]})
        return ok

    def fail(reason: str, rule: str) -> dict[str, Any]:
        return {"variant": variant, "title": VARIANTS[variant], "steps": steps,
                "outcome": {"ok": False, "failed_at": len(steps) - 1, "reason": reason, "rule": rule},
                "pkce": state.get("pkce"), "token": state.get("token")}

    rpc = compact({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                   "params": {"name": "read_file", "arguments": {"path": "notes.txt"},
                              "_meta": {"io.modelcontextprotocol/protocolVersion": "2026-07-28",
                                        "io.modelcontextprotocol/clientCapabilities": {}}}})
    mcp_headers = [["Content-Type", "application/json"], ["Accept", "application/json, text/event-stream"],
                   ["MCP-Protocol-Version", "2026-07-28"], ["Mcp-Method", "tools/call"], ["Mcp-Name", "read_file"]]
    prm_url = MCP + "/.well-known/oauth-protected-resource/mcp"

    # 1-2: the first request has no token.
    http("client", "mcp", "MCP request without a token", "POST", RESOURCE, mcp_headers, rpc, None)
    http("mcp", "client", "401 with resource_metadata", None, None,
         [["WWW-Authenticate", 'Bearer resource_metadata="' + prm_url + '", scope="files:read"']], None, 401,
         "Authorisation required: the header says where to read the server's metadata")
    state["phase"] = "discovery"
    # 3-4: protected-resource metadata (RFC 9728).
    http("client", "mcp", "GET protected-resource metadata", "GET", prm_url, [], None, None)
    prm = {"resource": RESOURCE, "authorization_servers": [AS], "scopes_supported": ["files:read"],
           "bearer_methods_supported": ["header"]}
    http("mcp", "client", "200 resource metadata", None, None, [["Content-Type", "application/json"]], prm, 200)
    # 5-6: authorisation-server metadata (RFC 8414), first well-known URL.
    http("client", "as", "GET authorisation-server metadata", "GET", AS + "/.well-known/oauth-authorization-server",
         [], None, None)
    asm: dict[str, Any] = {"issuer": EVIL_AS if variant == "issuer_mismatch" else AS,
                           "authorization_endpoint": AS + "/authorize", "token_endpoint": AS + "/token",
                           "registration_endpoint": AS + "/register", "response_types_supported": ["code"],
                           "grant_types_supported": ["authorization_code", "refresh_token"],
                           "code_challenge_methods_supported": ["S256"],
                           "client_id_metadata_document_supported": True,
                           "authorization_response_iss_parameter_supported": True}
    if variant == "no_pkce_support":
        del asm["code_challenge_methods_supported"]
    http("as", "client", "200 authorisation-server metadata", None, None, [["Content-Type", "application/json"]],
         asm, 200)
    if not check("client", "issuer in the metadata equals the issuer asked", asm["issuer"] == AS,
                 "issuer " + asm["issuer"] + " vs " + AS,
                 "The issuer value in the document MUST be identical to the issuer identifier used to construct the "
                 "well-known URL (RFC 8414 §3.3)"):
        return fail("metadata issuer does not match: the client must not use it", "RFC 8414 §3.3")
    methods = asm.get("code_challenge_methods_supported")
    if not check("client", "PKCE S256 supported", methods is not None and "S256" in methods,
                 "code_challenge_methods_supported = " + (compact(methods) if methods is not None else "absent"),
                 "If code_challenge_methods_supported is absent … MCP clients MUST refuse to proceed"):
        return fail("the authorisation server does not advertise PKCE: the client refuses to proceed",
                    "MCP 2026-07-28, Authorization Code Protection")
    state["issuer"] = AS
    state["phase"] = "registration"
    # 7: registration.
    if variant == "dcr":
        reg = {"client_name": "Example MCP client", "redirect_uris": [REDIRECT], "grant_types": ["authorization_code"],
               "response_types": ["code"], "token_endpoint_auth_method": "none", "application_type": "native"}
        http("client", "as", "POST /register (Dynamic Client Registration)", "POST", AS + "/register",
             [["Content-Type", "application/json"]], reg, None)
        client_id = "dcr-" + random_string(rng, 12)
        http("as", "client", "201 client_id", None, None, [["Content-Type", "application/json"]],
             {"client_id": client_id, "redirect_uris": [REDIRECT]}, 201)
    else:
        client_id = CLIENT_ID_URL
        check("client", "client_id is the URL of its metadata document", True, client_id,
              "Client ID Metadata Documents: an HTTPS URL as client_id")
    state["client_id"] = client_id
    state["phase"] = "authorisation"
    # 8: PKCE and state.
    verifier = random_string(rng, 43)
    challenge = pkce_challenge(verifier)
    st = random_string(rng, 16)
    state["pkce"] = {"verifier": verifier, "challenge": challenge, "sha256_hex": sha256_hex(verifier)}
    check("client", "make the PKCE pair; record state and the expected issuer", True,
          "verifier " + verifier + " → challenge " + challenge)
    q = [["response_type", "code"], ["client_id", client_id], ["redirect_uri", REDIRECT], ["scope", "files:read"],
         ["state", st]]
    if variant != "pkce_skipped":
        q += [["code_challenge", challenge], ["code_challenge_method", "S256"]]
    q += [["resource", RESOURCE]]
    auth_url = AS + "/authorize?" + form(q)
    http("client", "browser", "open the authorisation URL", None, auth_url, [], None, None)
    http("browser", "as", "GET /authorize", "GET", auth_url, [], None, None)
    if variant != "dcr":
        http("as", "client", "the AS fetches the client's metadata document", "GET", CLIENT_ID_URL, [], None, None)
        http("client", "as", "200 client metadata", None, None, [["Content-Type", "application/json"]],
             {"client_id": CLIENT_ID_URL, "client_name": "Example MCP client", "redirect_uris": [REDIRECT],
              "grant_types": ["authorization_code"], "response_types": ["code"],
              "token_endpoint_auth_method": "none"}, 200)
    if variant == "pkce_skipped":
        http("as", "browser", "400 invalid_request", None, None, [["Content-Type", "application/json"]],
             {"error": "invalid_request", "error_description": "code_challenge required"}, 400,
             "OAuth 2.1 requires PKCE for the authorization code grant")
        return fail("no PKCE challenge: the authorisation server refuses", "OAuth 2.1 §4.1.1")
    steps.append({"seq": len(steps), "kind": "user", "from": "browser", "to": "as", "label": "the user signs in and consents",
                  "ok": True, "phase": state["phase"]})
    code = "code-" + random_string(rng, 12)
    iss = EVIL_AS if variant == "iss_mismatch" else AS
    cb = REDIRECT + "?" + form([["code", code], ["state", st], ["iss", iss]])
    http("as", "browser", "302 back to the client with code, state and iss", None, None, [["Location", cb]], None, 302)
    http("browser", "client", "GET the callback", "GET", cb, [], None, None)
    check("client", "state matches", True, st)
    if not check("client", "iss equals the recorded issuer", iss == state["issuer"], iss + " vs " + state["issuer"],
                 "MCP clients MUST validate a present iss against the recorded issuer before redeeming the code "
                 "(RFC 9207)"):
        return fail("iss does not match the issuer the client started with: possible mix-up, the code is not redeemed",
                    "RFC 9207 §2.4")
    state["phase"] = "token"
    # Token request.
    sent_verifier = verifier if variant != "wrong_verifier" else random_string(rng, 43)
    tok_body = form([["grant_type", "authorization_code"], ["code", code], ["redirect_uri", REDIRECT],
                     ["client_id", client_id], ["code_verifier", sent_verifier], ["resource", RESOURCE]])
    http("client", "as", "POST /token with the code verifier", "POST", AS + "/token",
         [["Content-Type", "application/x-www-form-urlencoded"]], tok_body, None)
    recomputed = pkce_challenge(sent_verifier)
    if not check("as", "BASE64URL(SHA-256(verifier)) equals the challenge", recomputed == challenge,
                 recomputed + " vs " + challenge, "RFC 7636 §4.6"):
        http("as", "client", "400 invalid_grant", None, None, [["Content-Type", "application/json"]],
             {"error": "invalid_grant", "error_description": "PKCE verification failed"}, 400)
        return fail("the verifier does not hash to the challenge: the code cannot be redeemed", "RFC 7636 §4.6")
    scope = "files:read"
    token = "at-" + random_string(rng, 20)
    aud = RESOURCE
    claims = {"iss": AS, "sub": "user-42", "aud": aud, "scope": scope, "client_id": client_id, "exp": 3600}
    state["token"] = {"value": token, "claims": claims}
    http("as", "client", "200 access token (audience = the MCP server)", None, None,
         [["Content-Type", "application/json"]],
         {"access_token": token, "token_type": "Bearer", "expires_in": 3600, "scope": scope}, 200)
    state["phase"] = "authorised"
    # Use it.
    presented = token
    presented_claims = claims
    if variant == "wrong_audience":
        presented = "at-" + random_string(rng, 20)
        presented_claims = dict(claims)
        presented_claims["aud"] = OTHER_RESOURCE
        state["token"]["presented"] = {"value": presented, "claims": presented_claims}
    call_headers = [["Authorization", "Bearer " + presented]] + mcp_headers
    tool = "write_file" if variant == "insufficient_scope" else "read_file"
    body = rpc if tool == "read_file" else rpc.replace("read_file", "write_file")
    call_headers = [h if h[0] != "Mcp-Name" else ["Mcp-Name", tool] for h in call_headers]
    http("client", "mcp", "MCP request with the bearer token", "POST", RESOURCE, call_headers, body, None)
    if not check("mcp", "token audience is this server", presented_claims["aud"] == RESOURCE,
                 "aud " + presented_claims["aud"] + " vs " + RESOURCE,
                 "MCP servers MUST validate that access tokens were issued specifically for them"):
        http("mcp", "client", "401 invalid_token", None, None,
             [["WWW-Authenticate", 'Bearer error="invalid_token", error_description="audience mismatch", '
                                   'resource_metadata="' + prm_url + '"']], None, 401)
        return fail("the token was minted for another server: rejected", "RFC 8707 §2; MCP Token Handling")
    need = "files:write" if tool == "write_file" else "files:read"
    if not check("mcp", "token scope covers " + tool, need in presented_claims["scope"].split(" "),
                 "scope " + presented_claims["scope"] + ", needs " + need):
        http("mcp", "client", "403 insufficient_scope", None, None,
             [["WWW-Authenticate", 'Bearer error="insufficient_scope", scope="files:write", resource_metadata="'
                                   + prm_url + '"']], None, 403, "The client can step up: ask again for files:read files:write")
        return fail("the token lacks files:write: 403, and the client may step up", "MCP Scope Challenge Handling")
    if variant == "token_passthrough":
        http("mcp", "upstream", "the server forwards the same token upstream", "GET", UPSTREAM + "/files/notes.txt",
             [["Authorization", "Bearer " + presented]], None, None)
        check("upstream", "token audience is this API", False, "aud " + RESOURCE + " vs " + UPSTREAM,
              "The MCP server MUST NOT pass through the token it received from the MCP client")
        http("upstream", "mcp", "401 invalid_token", None, None, [["WWW-Authenticate", 'Bearer error="invalid_token"']],
             None, 401)
        return fail("token passthrough: the upstream API rejects a token that was not issued for it",
                    "MCP 2026-07-28, Access Token Privilege Restriction")
    result = compact({"jsonrpc": "2.0", "id": 1, "result": {"content": [{"type": "text", "text": "meeting at 10"}],
                                                            "isError": False, "resultType": "complete"}})
    http("mcp", "client", "200 MCP response", None, None, [["Content-Type", "application/json"]], result, 200)
    return {"variant": variant, "title": VARIANTS[variant], "steps": steps,
            "outcome": {"ok": True, "failed_at": None, "reason": "authorised", "rule": ""},
            "pkce": state["pkce"], "token": state["token"]}
