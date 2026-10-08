/** The protocols module (engine 1.2.0, extended in 1.3.0): MCP in both eras, transports, OAuth 2.1, A2A,
 * a gateway, three attacks and their defences, views. */
export * from "./mcp";
export { TRANSPORTS, WORK_MS, HOST, SESSION_ID, frameSession, streamDrop } from "./transport";
export {
  VARIANTS, RESOURCE, AS, OTHER_RESOURCE, EVIL_AS, CLIENT_ID_URL, REDIRECT, UPSTREAM, UNRESERVED,
  sha256Hex, pkceChallenge, randomString, pct, form, runOauth,
} from "./oauth";
export { sha256, hex, b64url, utf8Bytes } from "./sha256";
export { integrationFrames, messageLabel, sequenceFrames, negotiation, journey, threeWays, FEATURES, a2aFrames, a2aSummary, flowFrames } from "./views";
export { SCENARIOS, ENGINE_SCENARIOS, protocolScenario } from "./scenarios";
export * as a2a from "./a2a";
export { POLICIES, SERVERS as GATEWAY_SERVERS, ALLOW, merge as gatewayMerge, gatewayRun } from "./gateway";
export { ATTACKS, NOTES_TOOL, HIDDEN, SECRET, toolHash, runSecurity } from "./security";
