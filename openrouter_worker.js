// ── Allowed origins ──
const ALLOWED_ORIGINS = [
  "https://mindcoach-ai.tiiny.site",
  "http://localhost:8000",
  "http://127.0.0.1:8000",
  "null",
  "",
];

// ── Rate limiting (per IP, in-memory best-effort) ──
const RATE_LIMIT_MAX = 60;
const RATE_LIMIT_WINDOW_MS = 10 * 60 * 1000;
const _rateLimitStore = new Map();

function checkRateLimit(ip) {
  const now = Date.now();
  const entry = _rateLimitStore.get(ip);
  if (!entry || now - entry.windowStart > RATE_LIMIT_WINDOW_MS) {
    _rateLimitStore.set(ip, { windowStart: now, count: 1 });
    return true;
  }
  entry.count++;
  return entry.count <= RATE_LIMIT_MAX;
}

// ── Provider stats (circuit breaker) — persisted in KV, bound as
// PROVIDER_STATE. Degrades gracefully (no crash, just no memory) if the
// binding isn't set up. ──
const STATS_KEY = "provider_stats_v1";
const FAILURE_COOLDOWN_MS = 5 * 60 * 1000;

async function loadStats(env) {
  if (!env.PROVIDER_STATE) return {};
  try {
    const raw = await env.PROVIDER_STATE.get(STATS_KEY);
    return raw ? JSON.parse(raw) : {};
  } catch {
    return {};
  }
}

async function saveStats(env, stats) {
  if (!env.PROVIDER_STATE) return;
  try {
    await env.PROVIDER_STATE.put(STATS_KEY, JSON.stringify(stats), {
      expirationTtl: 86400,
    });
  } catch {
    // non-fatal
  }
}

function isOnCooldown(stats, name) {
  const s = stats[name];
  return !!(s && s.failedUntil && s.failedUntil > Date.now());
}

function recordSuccess(stats, name, latencyMs) {
  const prev = stats[name]?.avgLatencyMs;
  stats[name] = {
    avgLatencyMs: prev ? Math.round(prev * 0.7 + latencyMs * 0.3) : latencyMs,
    failedUntil: 0,
  };
}

function recordFailure(stats, name) {
  const prev = stats[name] || {};
  stats[name] = { avgLatencyMs: prev.avgLatencyMs || 2500, failedUntil: Date.now() + FAILURE_COOLDOWN_MS };
}

function toPlainMessages(messages, systemPrompt) {
  const out = [];
  if (systemPrompt) out.push({ role: "system", content: systemPrompt });
  for (const msg of messages) {
    out.push({
      role: msg.role,
      content: typeof msg.content === "string" ? msg.content : msg.content?.[0]?.text || "",
    });
  }
  return out;
}

// ── SSE parsing helper ──
// Reads a byte stream, splits it into "\n\n"-delimited SSE events, and
// yields the raw event text. Provider-specific code then pulls the text
// delta out of each event's JSON payload.
async function* sseEvents(reader, signal) {
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      if (signal.aborted) return;
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buffer.indexOf("\n\n")) !== -1) {
        const rawEvent = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        yield rawEvent;
      }
    }
  } finally {
    try {
      reader.cancel();
    } catch {}
  }
}

function dataLine(rawEvent) {
  const line = rawEvent.split("\n").find((l) => l.startsWith("data:"));
  return line ? line.slice(5).trim() : null;
}

// ── Per-provider streaming generators ──
// Each yields { text } chunks as they arrive, or throws on setup failure
// (bad key, HTTP error, etc). No internal retries — the race in the main
// handler is what picks a winner across providers.

async function* streamOpenAIStyle(url, headers, bodyObj, signal) {
  const res = await fetch(url, {
    method: "POST",
    headers,
    body: JSON.stringify({ ...bodyObj, stream: true }),
    signal,
  });
  if (!res.ok) {
    const errData = await res.json().catch(() => null);
    throw new Error(errData?.error?.message || `HTTP ${res.status}`);
  }
  for await (const rawEvent of sseEvents(res.body.getReader(), signal)) {
    const payload = dataLine(rawEvent);
    if (!payload) continue;
    if (payload === "[DONE]") return;
    try {
      const json = JSON.parse(payload);
      const text = json?.choices?.[0]?.delta?.content;
      if (text) yield { text };
    } catch {
      // partial/malformed chunk — skip it, next one usually recovers
    }
  }
}

function streamGroq(messages, systemPrompt, apiKey, signal) {
  return streamOpenAIStyle(
    "https://api.groq.com/openai/v1/chat/completions",
    { "Content-Type": "application/json", Authorization: `Bearer ${apiKey}` },
    {
      model: "llama-3.3-70b-versatile",
      messages: toPlainMessages(messages, systemPrompt),
      max_tokens: 700,
      temperature: 0.5,
    },
    signal,
  );
}

function streamOpenRouter(messages, systemPrompt, apiKey, signal) {
  return streamOpenAIStyle(
    "https://openrouter.ai/v1/chat/completions",
    {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
      "HTTP-Referer": "https://mindcoach-ai.tiiny.site",
      "X-Title": "MindCoach AI",
    },
    {
      model: "openrouter/free",
      messages: toPlainMessages(messages, systemPrompt),
      max_tokens: 280,
      temperature: 0.45,
      top_p: 0.9,
      frequency_penalty: 0.1,
    },
    signal,
  );
}

function streamHuggingFace(messages, systemPrompt, apiKey, signal) {
  return streamOpenAIStyle(
    "https://router.huggingface.co/v1/chat/completions",
    { "Content-Type": "application/json", Authorization: `Bearer ${apiKey}` },
    {
      model: "meta-llama/Llama-3.1-8B-Instruct:auto",
      messages: toPlainMessages(messages, systemPrompt),
      max_tokens: 700,
      temperature: 0.5,
    },
    signal,
  );
}

async function* streamGemini(messages, systemPrompt, apiKey, signal) {
  const contents = messages
    .filter((m) => m.role === "user" || m.role === "assistant")
    .map((m) => ({
      role: m.role === "assistant" ? "model" : "user",
      parts: [{ text: typeof m.content === "string" ? m.content : m.content?.[0]?.text || "" }],
    }));
  const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse&key=${apiKey}`;
  const payload = { contents, generationConfig: { maxOutputTokens: 700, temperature: 0.6 } };
  if (systemPrompt) payload.systemInstruction = { parts: [{ text: systemPrompt }] };

  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });
  if (!res.ok) {
    const errData = await res.json().catch(() => null);
    throw new Error(errData?.error?.message || `HTTP ${res.status}`);
  }
  for await (const rawEvent of sseEvents(res.body.getReader(), signal)) {
    const payloadStr = dataLine(rawEvent);
    if (!payloadStr) continue;
    try {
      const json = JSON.parse(payloadStr);
      const text = json?.candidates?.[0]?.content?.parts?.[0]?.text;
      if (text) yield { text };
    } catch {}
  }
}

async function* streamWorkersAI(messages, systemPrompt, aiBinding, signal) {
  const stream = await aiBinding.run("@cf/meta/llama-4-scout-17b-16e-instruct", {
    messages: toPlainMessages(messages, systemPrompt),
    max_tokens: 700,
    stream: true,
  });
  for await (const rawEvent of sseEvents(stream.getReader(), signal)) {
    const payload = dataLine(rawEvent);
    if (!payload) continue;
    if (payload === "[DONE]") return;
    try {
      const json = JSON.parse(payload);
      if (json?.response) yield { text: json.response };
    } catch {}
  }
}

const PROVIDER_LABELS = {
  groq: "llama-3.3-70b-versatile (Groq)",
  workersai: "@cf/meta/llama-4-scout-17b-16e-instruct (Workers AI)",
  gemini: "gemini-2.5-flash",
  openrouter: "openrouter/free",
  huggingface: "meta-llama/Llama-3.1-8B-Instruct (Hugging Face)",
};

const PER_PROVIDER_TIMEOUT_MS = 8000;

export default {
  async fetch(request, env, ctx) {
    const origin = request.headers.get("Origin");
    const normalizedOrigin = origin ? origin.toLowerCase().trim() : "";
    const originAllowed = ALLOWED_ORIGINS.includes(normalizedOrigin);
    const requestUrl = new URL(request.url);
    const requestPath = requestUrl.pathname;

    const corsHeaders = {
      "Access-Control-Allow-Origin": originAllowed ? origin || "null" : "null",
      "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type, x-api-key, anthropic-version",
    };

    if (request.method === "OPTIONS") return new Response(null, { headers: corsHeaders });

    if (request.method === "GET") {
      return new Response(
        JSON.stringify({ status: "ok", message: "Worker is deployed and reachable.", path: requestPath }),
        { status: 200, headers: { ...corsHeaders, "Content-Type": "application/json" } },
      );
    }

    if (request.method !== "POST")
      return new Response("Method not allowed", { status: 405, headers: corsHeaders });

    if (requestPath !== "/") {
      return new Response(
        JSON.stringify({ error: { message: `Worker reached, but path not supported: ${requestPath}` } }),
        { status: 404, headers: { ...corsHeaders, "Content-Type": "application/json" } },
      );
    }

    if (!originAllowed) {
      return new Response(JSON.stringify({ error: { message: "Origin not allowed" } }), {
        status: 403,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    const ip = request.headers.get("CF-Connecting-IP") || "unknown";
    if (!checkRateLimit(ip)) {
      return new Response(
        JSON.stringify({ error: { message: "Rate limit exceeded. Please wait a few minutes and try again." } }),
        { status: 429, headers: { ...corsHeaders, "Content-Type": "application/json" } },
      );
    }

    const hasAnyProvider =
      env.GEMINI_API_KEY || env.GROQ_API_KEY || env.OPENROUTER_API_KEY || env.HF_API_KEY || env.AI;
    if (!hasAnyProvider) {
      return new Response(
        JSON.stringify({
          error: {
            message:
              "Server misconfigured: no AI provider is configured. Set at least one of GEMINI_API_KEY, GROQ_API_KEY, OPENROUTER_API_KEY, HF_API_KEY, or bind Workers AI.",
          },
        }),
        { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } },
      );
    }

    let body;
    try {
      body = await request.json();
    } catch (e) {
      return new Response(JSON.stringify({ error: { message: "Invalid JSON body" } }), {
        status: 400,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    const messages = body.messages || [];
    const systemPrompt = body.system || "";
    const stats = await loadStats(env);
    const requestStart = Date.now();

    // Build one streaming generator per configured, non-cooldown provider.
    const build = (name, makeGen) => ({
      name,
      controller: new AbortController(),
      makeGen,
    });

    let attempts = [];
    if (env.GROQ_API_KEY)
      attempts.push(build("groq", (signal) => streamGroq(messages, systemPrompt, env.GROQ_API_KEY, signal)));
    if (env.AI) attempts.push(build("workersai", (signal) => streamWorkersAI(messages, systemPrompt, env.AI, signal)));
    if (env.GEMINI_API_KEY)
      attempts.push(build("gemini", (signal) => streamGemini(messages, systemPrompt, env.GEMINI_API_KEY, signal)));
    if (env.OPENROUTER_API_KEY)
      attempts.push(
        build("openrouter", (signal) => streamOpenRouter(messages, systemPrompt, env.OPENROUTER_API_KEY, signal)),
      );
    if (env.HF_API_KEY)
      attempts.push(
        build("huggingface", (signal) => streamHuggingFace(messages, systemPrompt, env.HF_API_KEY, signal)),
      );

    let candidates = attempts.filter((a) => !isOnCooldown(stats, a.name));
    if (candidates.length === 0) candidates = attempts; // all on cooldown — try anyway

    // Fire everyone in parallel; whoever produces the first real text
    // chunk wins. Losers get aborted immediately to stop burning quota.
    const runners = candidates.map((c) => {
      const timeoutId = setTimeout(() => c.controller.abort(), PER_PROVIDER_TIMEOUT_MS);
      const iterator = c.makeGen(c.controller.signal)[Symbol.asyncIterator]();
      return { ...c, iterator, timeoutId };
    });

    const pending = new Map();
    for (const r of runners) {
      pending.set(
        r.name,
        r.iterator
          .next()
          .then((result) => ({ name: r.name, result }))
          .catch((error) => ({ name: r.name, error })),
      );
    }

    let winner = null;
    let winnerFirstText = null;

    while (pending.size > 0 && !winner) {
      const outcome = await Promise.race(pending.values());
      pending.delete(outcome.name);
      const runner = runners.find((r) => r.name === outcome.name);

      if (outcome.error) {
        recordFailure(stats, outcome.name);
        clearTimeout(runner.timeoutId);
        console.warn("Provider failed before first token:", outcome.name, outcome.error.message);
        continue;
      }
      const { result } = outcome;
      if (result.done) {
        // Stream ended without ever yielding text — treat as a failure.
        recordFailure(stats, outcome.name);
        clearTimeout(runner.timeoutId);
        continue;
      }
      if (result.value && result.value.text) {
        winner = runner;
        winnerFirstText = result.value.text;
        break;
      }
      // Empty delta (e.g. a role-only event) — keep listening to this one.
      pending.set(
        outcome.name,
        runner.iterator
          .next()
          .then((r) => ({ name: outcome.name, result: r }))
          .catch((error) => ({ name: outcome.name, error })),
      );
    }

    // Abort every non-winning attempt now — no point letting them keep
    // generating tokens nobody will see.
    for (const r of runners) {
      if (!winner || r.name !== winner.name) {
        r.controller.abort();
        clearTimeout(r.timeoutId);
      }
    }

    if (!winner) {
      ctx.waitUntil(saveStats(env, stats));
      return new Response(
        JSON.stringify({
          error: { message: "Coach is having trouble connecting right now. Give it a moment and try again." },
        }),
        { status: 502, headers: { ...corsHeaders, "Content-Type": "application/json" } },
      );
    }

    recordSuccess(stats, winner.name, Date.now() - requestStart);
    ctx.waitUntil(saveStats(env, stats));

    // Stream the winner's tokens to the client as newline-delimited JSON:
    // {"delta":"..."} for each chunk, then {"done":true,"model":"..."}.
    const encoder = new TextEncoder();
    const modelUsed = PROVIDER_LABELS[winner.name] || winner.name;

    const outStream = new ReadableStream({
      async start(controller) {
        controller.enqueue(encoder.encode(JSON.stringify({ delta: winnerFirstText }) + "\n"));
        try {
          while (true) {
            const { value, done } = await winner.iterator.next();
            if (done) break;
            if (value && value.text) {
              controller.enqueue(encoder.encode(JSON.stringify({ delta: value.text }) + "\n"));
            }
          }
        } catch (err) {
          console.warn("Winner stream broke mid-way:", winner.name, err.message);
          // Client still gets everything sent so far, plus the done event
          // below, so the UI can stop its typing indicator cleanly.
        }
        controller.enqueue(encoder.encode(JSON.stringify({ done: true, model: modelUsed }) + "\n"));
        controller.close();
      },
      cancel() {
        winner.controller.abort();
      },
    });

    return new Response(outStream, {
      status: 200,
      headers: { ...corsHeaders, "Content-Type": "application/x-ndjson; charset=utf-8" },
    });
  },
};
