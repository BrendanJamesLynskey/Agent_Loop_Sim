/**
 * MCP authorisation (OAuth 2.1, MCP 2026-07-28) as a state machine with mis-configured variants:
 * a port of agent_loop_sim/protocols/oauth.py, statement for statement.
 */
import data from "../protocols_data.json";
import { Rng } from "../rng";
import { compact, type Obj } from "./mcp";
import { b64url, hex, sha256, utf8Bytes } from "./sha256";

export const MCP = "https://mcp.example.com";
export const RESOURCE = MCP + "/mcp";
export const AS = "https://auth.example.com";
export const OTHER_RESOURCE = "https://calendar.example.com/mcp";
export const EVIL_AS = "https://evil.example.com";
export const CLIENT_ID_URL = "https://client.example.com/oauth/client.json";
export const REDIRECT = "http://127.0.0.1:33418/callback";
export const UPSTREAM = "https://api.files.example.com";
export const UNRESERVED = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~";
export const VARIANTS: Record<string, string> = data.oauth_variants as Record<string, string>;

export function sha256Hex(s: string): string {
  return hex(sha256(utf8Bytes(s)));
}

/** RFC 7636 S256: BASE64URL-ENCODE(SHA256(ASCII(code_verifier))). */
export function pkceChallenge(verifier: string): string {
  return b64url(sha256(utf8Bytes(verifier)));
}

export function randomString(rng: Rng, n: number): string {
  let out = "";
  for (let i = 0; i < n; i++) out += UNRESERVED[rng.randint(0, UNRESERVED.length - 1)]!;
  return out;
}

/** Percent-encode everything but the RFC 3986 unreserved characters (UTF-8, upper-case hex). */
export function pct(s: string): string {
  let out = "";
  for (const b of utf8Bytes(s)) {
    const c = String.fromCharCode(b);
    if (b < 0x80 && UNRESERVED.includes(c)) out += c;
    else out += "%" + b.toString(16).toUpperCase().padStart(2, "0");
  }
  return out;
}

export function form(pairs: string[][]): string {
  return pairs.map(([k, v]) => pct(k!) + "=" + pct(v!)).join("&");
}

export function runOauth(variant = "ok", seed = 7): Obj {
  if (!(variant in VARIANTS)) throw new Error("unknown variant " + variant);
  const rng = new Rng(seed);
  const steps: Obj[] = [];
  const state: Obj = { phase: "unauthenticated" };

  const http = (
    frm: string,
    to: string,
    label: string,
    method: string | null,
    url: string | null,
    headers: string[][],
    body: unknown,
    status: number | null,
    note = "",
  ): Obj => {
    const s: Obj = {
      seq: steps.length,
      kind: "http",
      from: frm,
      to,
      label,
      method,
      url,
      headers,
      body,
      status,
      note,
      ok: status === null || status < 400,
      phase: state.phase,
    };
    steps.push(s);
    return s;
  };

  const check = (actor: string, label: string, ok: boolean, detail: string, rule = ""): boolean => {
    steps.push({ seq: steps.length, kind: "check", from: actor, to: actor, label, ok, detail, rule, phase: state.phase });
    return ok;
  };

  const fail = (reason: string, rule: string): Obj => ({
    variant,
    title: VARIANTS[variant],
    steps,
    outcome: { ok: false, failed_at: steps.length - 1, reason, rule },
    pkce: state.pkce ?? null,
    token: state.token ?? null,
  });

  const rpc = compact({
    jsonrpc: "2.0",
    id: 1,
    method: "tools/call",
    params: {
      name: "read_file",
      arguments: { path: "notes.txt" },
      _meta: { "io.modelcontextprotocol/protocolVersion": "2026-07-28", "io.modelcontextprotocol/clientCapabilities": {} },
    },
  });
  const mcpHeaders = [
    ["Content-Type", "application/json"],
    ["Accept", "application/json, text/event-stream"],
    ["MCP-Protocol-Version", "2026-07-28"],
    ["Mcp-Method", "tools/call"],
    ["Mcp-Name", "read_file"],
  ];
  const prmUrl = MCP + "/.well-known/oauth-protected-resource/mcp";

  http("client", "mcp", "MCP request without a token", "POST", RESOURCE, mcpHeaders, rpc, null);
  http(
    "mcp",
    "client",
    "401 with resource_metadata",
    null,
    null,
    [["WWW-Authenticate", 'Bearer resource_metadata="' + prmUrl + '", scope="files:read"']],
    null,
    401,
    "Authorisation required: the header says where to read the server's metadata",
  );
  state.phase = "discovery";
  http("client", "mcp", "GET protected-resource metadata", "GET", prmUrl, [], null, null);
  const prm = {
    resource: RESOURCE,
    authorization_servers: [AS],
    scopes_supported: ["files:read"],
    bearer_methods_supported: ["header"],
  };
  http("mcp", "client", "200 resource metadata", null, null, [["Content-Type", "application/json"]], prm, 200);
  http("client", "as", "GET authorisation-server metadata", "GET", AS + "/.well-known/oauth-authorization-server", [], null, null);
  const asm: Obj = {
    issuer: variant === "issuer_mismatch" ? EVIL_AS : AS,
    authorization_endpoint: AS + "/authorize",
    token_endpoint: AS + "/token",
    registration_endpoint: AS + "/register",
    response_types_supported: ["code"],
    grant_types_supported: ["authorization_code", "refresh_token"],
    code_challenge_methods_supported: ["S256"],
    client_id_metadata_document_supported: true,
    authorization_response_iss_parameter_supported: true,
  };
  if (variant === "no_pkce_support") delete asm.code_challenge_methods_supported;
  http("as", "client", "200 authorisation-server metadata", null, null, [["Content-Type", "application/json"]], asm, 200);
  if (
    !check(
      "client",
      "issuer in the metadata equals the issuer asked",
      asm.issuer === AS,
      "issuer " + asm.issuer + " vs " + AS,
      "The issuer value in the document MUST be identical to the issuer identifier used to construct the " +
        "well-known URL (RFC 8414 §3.3)",
    )
  ) {
    return fail("metadata issuer does not match: the client must not use it", "RFC 8414 §3.3");
  }
  const methods = asm.code_challenge_methods_supported ?? null;
  if (
    !check(
      "client",
      "PKCE S256 supported",
      methods !== null && methods.includes("S256"),
      "code_challenge_methods_supported = " + (methods !== null ? compact(methods) : "absent"),
      "If code_challenge_methods_supported is absent … MCP clients MUST refuse to proceed",
    )
  ) {
    return fail(
      "the authorisation server does not advertise PKCE: the client refuses to proceed",
      "MCP 2026-07-28, Authorization Code Protection",
    );
  }
  state.issuer = AS;
  state.phase = "registration";
  let clientId: string;
  if (variant === "dcr") {
    const reg = {
      client_name: "Example MCP client",
      redirect_uris: [REDIRECT],
      grant_types: ["authorization_code"],
      response_types: ["code"],
      token_endpoint_auth_method: "none",
      application_type: "native",
    };
    http("client", "as", "POST /register (Dynamic Client Registration)", "POST", AS + "/register", [["Content-Type", "application/json"]], reg, null);
    clientId = "dcr-" + randomString(rng, 12);
    http("as", "client", "201 client_id", null, null, [["Content-Type", "application/json"]], { client_id: clientId, redirect_uris: [REDIRECT] }, 201);
  } else {
    clientId = CLIENT_ID_URL;
    check("client", "client_id is the URL of its metadata document", true, clientId, "Client ID Metadata Documents: an HTTPS URL as client_id");
  }
  state.client_id = clientId;
  state.phase = "authorisation";
  const verifier = randomString(rng, 43);
  const challenge = pkceChallenge(verifier);
  const st = randomString(rng, 16);
  state.pkce = { verifier, challenge, sha256_hex: sha256Hex(verifier) };
  check("client", "make the PKCE pair; record state and the expected issuer", true, "verifier " + verifier + " → challenge " + challenge);
  let q = [
    ["response_type", "code"],
    ["client_id", clientId],
    ["redirect_uri", REDIRECT],
    ["scope", "files:read"],
    ["state", st],
  ];
  if (variant !== "pkce_skipped") q = q.concat([["code_challenge", challenge], ["code_challenge_method", "S256"]]);
  q = q.concat([["resource", RESOURCE]]);
  const authUrl = AS + "/authorize?" + form(q);
  http("client", "browser", "open the authorisation URL", null, authUrl, [], null, null);
  http("browser", "as", "GET /authorize", "GET", authUrl, [], null, null);
  if (variant !== "dcr") {
    http("as", "client", "the AS fetches the client's metadata document", "GET", CLIENT_ID_URL, [], null, null);
    http("client", "as", "200 client metadata", null, null, [["Content-Type", "application/json"]], {
      client_id: CLIENT_ID_URL,
      client_name: "Example MCP client",
      redirect_uris: [REDIRECT],
      grant_types: ["authorization_code"],
      response_types: ["code"],
      token_endpoint_auth_method: "none",
    }, 200);
  }
  if (variant === "pkce_skipped") {
    http("as", "browser", "400 invalid_request", null, null, [["Content-Type", "application/json"]], {
      error: "invalid_request",
      error_description: "code_challenge required",
    }, 400, "OAuth 2.1 requires PKCE for the authorization code grant");
    return fail("no PKCE challenge: the authorisation server refuses", "OAuth 2.1 §4.1.1");
  }
  steps.push({ seq: steps.length, kind: "user", from: "browser", to: "as", label: "the user signs in and consents", ok: true, phase: state.phase });
  const code = "code-" + randomString(rng, 12);
  const iss = variant === "iss_mismatch" ? EVIL_AS : AS;
  const cb = REDIRECT + "?" + form([["code", code], ["state", st], ["iss", iss]]);
  http("as", "browser", "302 back to the client with code, state and iss", null, null, [["Location", cb]], null, 302);
  http("browser", "client", "GET the callback", "GET", cb, [], null, null);
  check("client", "state matches", true, st);
  if (
    !check(
      "client",
      "iss equals the recorded issuer",
      iss === state.issuer,
      iss + " vs " + state.issuer,
      "MCP clients MUST validate a present iss against the recorded issuer before redeeming the code (RFC 9207)",
    )
  ) {
    return fail("iss does not match the issuer the client started with: possible mix-up, the code is not redeemed", "RFC 9207 §2.4");
  }
  state.phase = "token";
  const sentVerifier = variant !== "wrong_verifier" ? verifier : randomString(rng, 43);
  const tokBody = form([
    ["grant_type", "authorization_code"],
    ["code", code],
    ["redirect_uri", REDIRECT],
    ["client_id", clientId],
    ["code_verifier", sentVerifier],
    ["resource", RESOURCE],
  ]);
  http("client", "as", "POST /token with the code verifier", "POST", AS + "/token", [["Content-Type", "application/x-www-form-urlencoded"]], tokBody, null);
  const recomputed = pkceChallenge(sentVerifier);
  if (!check("as", "BASE64URL(SHA-256(verifier)) equals the challenge", recomputed === challenge, recomputed + " vs " + challenge, "RFC 7636 §4.6")) {
    http("as", "client", "400 invalid_grant", null, null, [["Content-Type", "application/json"]], {
      error: "invalid_grant",
      error_description: "PKCE verification failed",
    }, 400);
    return fail("the verifier does not hash to the challenge: the code cannot be redeemed", "RFC 7636 §4.6");
  }
  const scope = "files:read";
  const token = "at-" + randomString(rng, 20);
  const aud = RESOURCE;
  const claims: Obj = { iss: AS, sub: "user-42", aud, scope, client_id: clientId, exp: 3600 };
  state.token = { value: token, claims };
  http("as", "client", "200 access token (audience = the MCP server)", null, null, [["Content-Type", "application/json"]], {
    access_token: token,
    token_type: "Bearer",
    expires_in: 3600,
    scope,
  }, 200);
  state.phase = "authorised";
  let presented = token;
  let presentedClaims: Obj = claims;
  if (variant === "wrong_audience") {
    presented = "at-" + randomString(rng, 20);
    presentedClaims = { ...claims };
    presentedClaims.aud = OTHER_RESOURCE;
    state.token.presented = { value: presented, claims: presentedClaims };
  }
  let callHeaders = [["Authorization", "Bearer " + presented]].concat(mcpHeaders);
  const tool = variant === "insufficient_scope" ? "write_file" : "read_file";
  const body = tool === "read_file" ? rpc : rpc.split("read_file").join("write_file");
  callHeaders = callHeaders.map((h) => (h[0] !== "Mcp-Name" ? h : ["Mcp-Name", tool]));
  http("client", "mcp", "MCP request with the bearer token", "POST", RESOURCE, callHeaders, body, null);
  if (
    !check(
      "mcp",
      "token audience is this server",
      presentedClaims.aud === RESOURCE,
      "aud " + presentedClaims.aud + " vs " + RESOURCE,
      "MCP servers MUST validate that access tokens were issued specifically for them",
    )
  ) {
    http("mcp", "client", "401 invalid_token", null, null, [
      ["WWW-Authenticate", 'Bearer error="invalid_token", error_description="audience mismatch", resource_metadata="' + prmUrl + '"'],
    ], null, 401);
    return fail("the token was minted for another server: rejected", "RFC 8707 §2; MCP Token Handling");
  }
  const need = tool === "write_file" ? "files:write" : "files:read";
  if (!check("mcp", "token scope covers " + tool, presentedClaims.scope.split(" ").includes(need), "scope " + presentedClaims.scope + ", needs " + need)) {
    http("mcp", "client", "403 insufficient_scope", null, null, [
      ["WWW-Authenticate", 'Bearer error="insufficient_scope", scope="files:write", resource_metadata="' + prmUrl + '"'],
    ], null, 403, "The client can step up: ask again for files:read files:write");
    return fail("the token lacks files:write: 403, and the client may step up", "MCP Scope Challenge Handling");
  }
  if (variant === "token_passthrough") {
    http("mcp", "upstream", "the server forwards the same token upstream", "GET", UPSTREAM + "/files/notes.txt", [["Authorization", "Bearer " + presented]], null, null);
    check("upstream", "token audience is this API", false, "aud " + RESOURCE + " vs " + UPSTREAM, "The MCP server MUST NOT pass through the token it received from the MCP client");
    http("upstream", "mcp", "401 invalid_token", null, null, [["WWW-Authenticate", 'Bearer error="invalid_token"']], null, 401);
    return fail("token passthrough: the upstream API rejects a token that was not issued for it", "MCP 2026-07-28, Access Token Privilege Restriction");
  }
  const result = compact({
    jsonrpc: "2.0",
    id: 1,
    result: { content: [{ type: "text", text: "meeting at 10" }], isError: false, resultType: "complete" },
  });
  http("mcp", "client", "200 MCP response", null, null, [["Content-Type", "application/json"]], result, 200);
  return {
    variant,
    title: VARIANTS[variant],
    steps,
    outcome: { ok: true, failed_at: null, reason: "authorised", rule: "" },
    pkce: state.pkce,
    token: state.token,
  };
}
