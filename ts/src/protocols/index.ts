/** The protocols module (engine 1.2.0): MCP in both eras, transports, OAuth 2.1, views. */
export * from "./mcp";
export { TRANSPORTS, WORK_MS, HOST, SESSION_ID, frameSession, streamDrop } from "./transport";
export {
  VARIANTS, RESOURCE, AS, OTHER_RESOURCE, EVIL_AS, CLIENT_ID_URL, REDIRECT, UPSTREAM, UNRESERVED,
  sha256Hex, pkceChallenge, randomString, pct, form, runOauth,
} from "./oauth";
export { sha256, hex, b64url, utf8Bytes } from "./sha256";
export { integrationFrames, messageLabel, sequenceFrames, negotiation, journey, threeWays, FEATURES } from "./views";
export { SCENARIOS, ENGINE_SCENARIOS, protocolScenario } from "./scenarios";
