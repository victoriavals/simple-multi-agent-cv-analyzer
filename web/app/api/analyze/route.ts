export const runtime = "nodejs";

const MAX_REQUEST_BYTES = 5 * 1024 * 1024 + 128 * 1024;

function errorResponse(code: string, status: number) {
  return Response.json({ type: "error", code }, { status, headers: { "Cache-Control": "no-store" } });
}

export async function POST(request: Request) {
  const contentType = request.headers.get("content-type");
  if (!request.body || !contentType?.startsWith("multipart/form-data;")) {
    return errorResponse("invalid_form", 415);
  }
  if (Number(request.headers.get("content-length")) > MAX_REQUEST_BYTES) {
    return errorResponse("file_too_large", 413);
  }

  const apiUrl = process.env.CV_ANALYZER_API_URL || "http://127.0.0.1:8000";
  let size = 0;
  let tooLarge = false;
  const body = request.body.pipeThrough(new TransformStream<Uint8Array, Uint8Array>({
    transform(chunk, controller) {
      size += chunk.byteLength;
      if (size > MAX_REQUEST_BYTES) {
        tooLarge = true;
        controller.error(new Error("Upload limit exceeded"));
      } else {
        controller.enqueue(chunk);
      }
    },
  }));
  try {
    const options: RequestInit & { duplex: "half" } = {
      method: "POST",
      headers: { "content-type": contentType },
      body,
      duplex: "half",
      cache: "no-store",
      signal: AbortSignal.any([request.signal, AbortSignal.timeout(300_000)]),
    };
    const response = await fetch(`${apiUrl.replace(/\/$/, "")}/analyze`, options);
    return new Response(response.body, {
      status: response.status,
      headers: {
        "Content-Type": response.headers.get("content-type") || "application/json",
        "Cache-Control": "no-store",
        "X-Accel-Buffering": "no",
      },
    });
  } catch {
    if (tooLarge) return errorResponse("file_too_large", 413);
    return errorResponse("backend_unavailable", 503);
  }
}
