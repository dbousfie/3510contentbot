import { serve } from "https://deno.land/std@0.224.0/http/server.ts";

const OPENAI_API_KEY = Deno.env.get("OPENAI_API_KEY");
const OPENAI_MODEL = Deno.env.get("OPENAI_MODEL") || "gpt-4o-mini";
const SYLLABUS_LINK = Deno.env.get("SYLLABUS_LINK") || "";
const ADMIN_TOKEN = Deno.env.get("ADMIN_TOKEN") || "";
const RAG_TOP_K = parseInt(Deno.env.get("RAG_TOP_K") || "3");
const RAG_MIN_SCORE = parseFloat(Deno.env.get("RAG_MIN_SCORE") || "0.25");
const STRICT_RAG = (Deno.env.get("STRICT_RAG") || "true").toLowerCase() === "true";

const kv = await Deno.openKv();

function cors(body: string, status = 200) {
  return new Response(body, {
    status,
    headers: {
      "Content-Type": "text/plain",
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Admin-Token",
    },
  });
}

serve(async (req: Request): Promise<Response> => {
  const url = new URL(req.url);
  const path = url.pathname;

  if (req.method === "OPTIONS") return cors("", 204);

  if (path === "/chat") return handleChat(req);
  if (path === "/ingest") return handleIngest(req);
  if (path === "/wipe") return handleWipe(req);
  if (path === "/stats") return handleStats(req);

  return cors("Not Found", 404);
});

// --- Chat handler ---
async function handleChat(req: Request): Promise<Response> {
  let body: { query: string };
  try {
    body = await req.json();
  } catch {
    return cors("Invalid JSON", 400);
  }

  const query = body.query;
  if (!OPENAI_API_KEY) return cors("Missing OpenAI API key", 500);

  // Retrieve top-K matches
  const chunks = [];
  for await (const e of kv.list({ prefix: ["lec"] })) {
    chunks.push(e.value as any);
  }

  // naive cosine similarity placeholder — assume chunks already scored
  const ranked = chunks
    .map((c) => ({ ...c, score: Math.random() })) // replace with real sim search
    .sort((a, b) => b.score - a.score)
    .slice(0, RAG_TOP_K)
    .filter((c) => c.score >= RAG_MIN_SCORE);

  if (STRICT_RAG && ranked.length === 0) {
    return cors("I can’t find that in the course materials.");
  }

  const context = ranked.map((r) => r.text).join("\n\n");
  const sourceTitles = [...new Set(ranked.map((r) => r.title))];

  const messages = [
    { role: "system", content: "You are an accurate course assistant. Answer based only on the provided context and syllabus. Do not invent sources." },
    { role: "system", content: `Syllabus: ${SYLLABUS_LINK}` },
    { role: "system", content: `Context from lectures:\n${context}` },
    { role: "user", content: query },
  ];

  const openaiResp = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${OPENAI_API_KEY}`,
    },
    body: JSON.stringify({ model: OPENAI_MODEL, messages, max_tokens: 1200 }),
  });

  const json = await openaiResp.json();
  const base = json?.choices?.[0]?.message?.content || "No response";

  // --- enforce exact titles ---
  const cleaned = String(base).replace(/^\s*Sources:.*$/gmi, "").trim();
  const exactSources = sourceTitles.length
    ? `\n\nSources: ${sourceTitles.join("; ")}`
    : "";
  const reply = `${cleaned}${exactSources}\n\nThere may be errors in my responses; always refer to the course web page: ${SYLLABUS_LINK}`;

  return cors(reply);
}

// --- Ingest handler ---
async function handleIngest(req: Request): Promise<Response> {
  if (!checkAuth(req)) return cors("unauthorized", 401);

  const body = await req.json();
  const items = body.items || [];
  for (const it of items) {
    const key = ["lec", it.id];
    await kv.set(key, { id: it.id, title: it.title, text: it.text });
  }
  return cors("ok");
}

// --- Wipe handler ---
async function handleWipe(req: Request): Promise<Response> {
  if (!checkAuth(req)) return cors("unauthorized", 401);

  let deleted = 0;
  for await (const e of kv.list({ prefix: ["lec"] })) {
    await kv.delete(e.key);
    deleted++;
  }
  return cors(`wiped ${deleted} keys`);
}

// --- Stats handler ---
async function handleStats(req: Request): Promise<Response> {
  if (!checkAuth(req)) return cors("unauthorized", 401);

  let lectures = 0;
  const sample: string[] = [];
  for await (const e of kv.list({ prefix: ["lec"] })) {
    lectures++;
    if (sample.length < 5) sample.push((e.value as any).title);
  }
  return cors(JSON.stringify({ lectures, sample }));
}

// --- Auth helper ---
function checkAuth(req: Request): boolean {
  const h = req.headers.get("authorization") || req.headers.get("x-admin-token") || "";
  return h.replace(/^Bearer\s+/i, "").trim() === ADMIN_TOKEN.trim();
}
