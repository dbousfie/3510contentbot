// main.ts — RAG + admin + exact Sources + robust routing

import { serve } from "https://deno.land/std@0.224.0/http/server.ts";

// Env
const OPENAI_API_KEY       = Deno.env.get("OPENAI_API_KEY") ?? "";
const OPENAI_MODEL         = Deno.env.get("OPENAI_MODEL") ?? "gpt-4o-mini";
const EMBEDDING_MODEL      = Deno.env.get("EMBEDDING_MODEL") ?? "text-embedding-3-small";
const ADMIN_TOKEN          = (Deno.env.get("ADMIN_TOKEN") ?? "").trim();

const QUALTRICS_API_TOKEN  = Deno.env.get("QUALTRICS_API_TOKEN") ?? "";
const QUALTRICS_SURVEY_ID  = Deno.env.get("QUALTRICS_SURVEY_ID") ?? "";
const QUALTRICS_DATACENTER = Deno.env.get("QUALTRICS_DATACENTER") ?? "";
const SYLLABUS_LINK        = Deno.env.get("SYLLABUS_LINK") ?? "";

const STRICT_RAG = (Deno.env.get("STRICT_RAG") ?? "true").toLowerCase() === "true";
const MIN_SCORE  = Number(Deno.env.get("RAG_MIN_SCORE") ?? "0.28");
const TOP_K      = Number(Deno.env.get("RAG_TOP_K") ?? "3");

// CORS
const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Admin-Token",
};
const respond = (body: string, status = 200, ct = "text/plain") =>
  new Response(body, { status, headers: { ...CORS, "Content-Type": ct } });

// KV
const kv = await Deno.openKv();

// ---- helpers ----
function adminTokenFrom(req: Request): string {
  const auth = req.headers.get("authorization") ?? "";
  const alt  = req.headers.get("x-admin-token") ?? "";
  const bearer = /^Bearer\s+/i.test(auth) ? auth.replace(/^Bearer\s+/i, "") : "";
  return (bearer || alt).trim();
}
function requireAdmin(req: Request) {
  return ADMIN_TOKEN && adminTokenFrom(req) === ADMIN_TOKEN;
}
function cosine(a: number[], b: number[]) {
  let d=0, na=0, nb=0;
  for (let i=0;i<a.length;i++){ d+=a[i]*b[i]; na+=a[i]*a[i]; nb+=b[i]*b[i]; }
  return d/(Math.sqrt(na)*Math.sqrt(nb)+1e-8);
}
async function embed(text: string): Promise<number[]> {
  const r = await fetch("https://api.openai.com/v1/embeddings", {
    method: "POST",
    headers: { "Content-Type":"application/json", Authorization:`Bearer ${OPENAI_API_KEY}` },
    body: JSON.stringify({ model: EMBEDDING_MODEL, input: text }),
  });
  if (!r.ok) throw new Error(`embedding ${r.status}: ${await r.text()}`);
  const j = await r.json();
  return j.data[0].embedding as number[];
}
function footer() {
  return /^https?:\/\//i.test(SYLLABUS_LINK)
    ? `\n\nThere may be errors in my responses; always refer to the course web page: ${SYLLABUS_LINK}`
    : `\n\nThere may be errors in my responses; consult the official course page.`;
}

// ---- admin: /ingest ----
// Body: { items: [{ id, title, text }] }
async function handleIngest(req: Request) {
  if (!requireAdmin(req)) return respond("unauthorized", 401);
  let body: { items: { id: string; title: string; text: string }[] } = { items: [] };
  try { body = await req.json(); } catch { return respond("Invalid JSON", 400); }

  for (const it of body.items || []) {
    const parts = chunkByChars(it.text, 1700, 200);
    await kv.set(["lec", it.id, "meta"], { title: it.title, n: parts.length });
    for (let i=0;i<parts.length;i++){
      const text = parts[i];
      const e = await embed(text);
      await kv.set(["lec", it.id, "chunk", i], { text });
      await kv.set(["lec", it.id, "vec", i],   { e });
    }
  }
  return respond("ok");
}
function chunkByChars(s: string, max=1700, overlap=200) {
  const out:string[]=[]; for(let i=0;i<s.length;i+=max-overlap) out.push(s.slice(i,i+max)); return out;
}

// ---- admin: /retitle ----
// Body: { id, title }
async function handleRetitle(req: Request) {
  if (!requireAdmin(req)) return respond("unauthorized", 401);
  let body: { id?: string; title?: string } = {};
  try { body = await req.json(); } catch { return respond("Invalid JSON", 400); }
  if (!body.id || !body.title) return respond("id and title required", 400);
  const meta = await kv.get<{ title: string; n: number }>(["lec", body.id, "meta"]);
  if (!meta.value) return respond("not found", 404);
  await kv.set(["lec", body.id, "meta"], { ...meta.value, title: body.title });
  return respond("ok");
}

// ---- admin: /wipe ----
async function handleWipe(req: Request) {
  if (!requireAdmin(req)) return respond("unauthorized", 401);
  let n=0; for await (const e of kv.list({ prefix: ["lec"] })) { await kv.delete(e.key); n++; }
  return respond(`wiped ${n} keys`);
}

// ---- admin: /stats ----
async function handleStats(req: Request) {
  if (!requireAdmin(req)) return respond("unauthorized", 401);
  let lectures=0, chunks=0, vecs=0; const titles=new Set<string>();
  for await (const e of kv.list({ prefix:["lec"] })) {
    const k = e.key as Deno.KvKey;
    if (k.length===3 && k[2]==="meta") { lectures++; titles.add((e.value as any)?.title ?? ""); }
    if (k.length===4 && k[2]==="chunk") chunks++;
    if (k.length===4 && k[2]==="vec")   vecs++;
  }
  return respond(JSON.stringify({ lectures, chunks, vecs, sample: Array.from(titles).filter(Boolean).slice(0,10) }, null, 2), 200, "application/json");
}

// ---- /chat (and legacy "/") ----
// Body: { query }
async function handleChat(req: Request) {
  if (!OPENAI_API_KEY) return respond("Missing OpenAI API key", 500);
  let body: { query?: string } = {};
  try { body = await req.json(); } catch { return respond("Invalid JSON", 400); }
  const userQuery = (body.query || "").trim();
  if (!userQuery) return respond("Missing 'query' in body", 400);

  const syllabus = await Deno.readTextFile("syllabus.md").catch(()=>"Error loading syllabus.");

  // retrieve
  type Hit = { score:number; id:string; i:number; text:string; title:string };
  const hits: Hit[] = [];
  let retrievalFailed = false;
  try {
    const qv = await embed(userQuery);
    for await (const e of kv.list<{ e:number[] }>({ prefix: ["lec"] })) {
      const k = e.key as Deno.KvKey;
      if (k.length===4 && k[0]==="lec" && k[2]==="vec") {
        const id = String(k[1]); const i = Number(k[3]);
        const score = cosine(qv, e.value.e);
        const ch = await kv.get<{ text:string }>(["lec", id, "chunk", i]);
        const meta = await kv.get<{ title:string }>(["lec", id, "meta"]);
        if (!ch.value?.text || !meta.value?.title) continue;
        hits.push({ score, id, i, text: ch.value.text, title: meta.value.title });
      }
    }
  } catch { retrievalFailed = true; }

  hits.sort((a,b)=>b.score-a.score);
  const filtered = hits.filter(h=>h.score>=MIN_SCORE).slice(0,TOP_K);
  if (STRICT_RAG && !filtered.length && !retrievalFailed) {
    return respond(`I can’t find this in the course materials I have. Please check lecture titles or rephrase.${footer()}`);
  }
  const top = filtered.length ? filtered : hits.slice(0,TOP_K);
  const context = top.map((h,idx)=>`(${idx+1}) ${h.title}\n${h.text}`).join("\n\n---\n\n");
  const sourceTitles = Array.from(new Set(top.map(h=>h.title)));

  const messages = [
    { role: "system", content: STRICT_RAG
        ? "Answer ONLY using the CONTEXT and the syllabus text provided. If the answer is not in the CONTEXT/syllabus, say you don’t have that information. Do not speculate. Do not mention retrieval."
        : "Use the CONTEXT and syllabus when provided. Prefer them. Do not mention retrieval." },
    { role: "system", content: `Here is important context from syllabus.md:\n${syllabus}` },
    { role: "user",   content: context ? `QUESTION:\n${userQuery}\n\nCONTEXT:\n${context}` : userQuery },
  ] as const;

  const r = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: { "Content-Type":"application/json", Authorization:`Bearer ${OPENAI_API_KEY}` },
    body: JSON.stringify({ model: OPENAI_MODEL, messages, temperature: 0.2, max_tokens: 1500 }),
  });
  const j = await r.json();
  const base = j?.choices?.[0]?.message?.content || "No response";

  // enforce exact Sources (remove any model-written Sources:)
  const cleaned = String(base).replace(/^\s*Sources:.*$/gmi, "").trim();
  const exactSources = sourceTitles.length ? `\n\nSources: ${sourceTitles.join("; ")}` : "";
  let reply = cleaned + exactSources;

  // Qualtrics (optional)
  let qStatus = "Qualtrics not called";
  if (QUALTRICS_API_TOKEN && QUALTRICS_SURVEY_ID && QUALTRICS_DATACENTER) {
    try {
      const qt = await fetch(`https://${QUALTRICS_DATACENTER}.qualtrics.com/API/v3/surveys/${QUALTRICS_SURVEY_ID}/responses`, {
        method: "POST",
        headers: { "Content-Type":"application/json", "X-API-TOKEN": QUALTRICS_API_TOKEN },
        body: JSON.stringify({ values: { responseText: reply + footer(), queryText: userQuery } }),
      });
      qStatus = `Qualtrics status: ${qt.status}`;
    } catch (e) { qStatus = `Qualtrics error: ${(e as Error).message}`; }
  }

  return respond(`${reply}${footer()}\n<!-- ${qStatus} -->`);
}

// ---- router ----
serve(async (req) => {
  const path = new URL(req.url).pathname;
  if (req.method === "OPTIONS") return new Response(null, { status: 204, headers: CORS });
  if (req.method !== "POST")   return respond("Method Not Allowed", 405);

  if (path === "/ingest")  return handleIngest(req);
  if (path === "/retitle") return handleRetitle(req);
  if (path === "/wipe")    return handleWipe(req);
  if (path === "/stats")   return handleStats(req);
  if (path === "/chat")    return handleChat(req);
  if (path === "/")        return handleChat(req); // legacy root
  return handleChat(req);
});
