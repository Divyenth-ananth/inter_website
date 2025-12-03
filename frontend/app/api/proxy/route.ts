export const runtime = 'nodejs';

import { NextRequest } from 'next/server';

export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData();

    const ngrokUrl = process.env.NGROK_URL || 'https://team37.ngrok.io/v1/vqa/single';
    const username = process.env.NGROK_USER || 'isro';
    const password = process.env.NGROK_PASS || 'OneEarth';
    const authHeader = 'Basic ' + Buffer.from(`${username}:${password}`).toString('base64');

    const ngrokRes = await fetch(ngrokUrl, {
      method: 'POST',
      body: formData as any,
      headers: {
        Authorization: authHeader,
        'ngrok-skip-browser-warning': 'true',
      },
    });

    const contentType = ngrokRes.headers.get('content-type') || 'application/json';
    const buffer = await ngrokRes.arrayBuffer();

    return new Response(Buffer.from(buffer), {
      status: ngrokRes.status,
      headers: {
        'content-type': contentType,
      },
    });
  } catch (err) {
    console.error('Proxy error:', err);
    return new Response(JSON.stringify({ error: String(err) }), { status: 500, headers: { 'content-type': 'application/json' } });
  }
}
