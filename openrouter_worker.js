export default {
  async fetch(request, env, ctx) {
    const corsHeaders = {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, x-api-key, anthropic-version',
    };

    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: corsHeaders });
    }

    if (request.method !== 'POST') {
      return new Response('Method not allowed', { status: 405 });
    }

    try {
      const body = await request.json();

      const messages = body.messages || [];
      const systemPrompt = body.system || '';

      // Build messages array for OpenRouter
      const openRouterMessages = [];
      
      if (systemPrompt) {
        openRouterMessages.push({ role: 'system', content: systemPrompt });
      }

      for (const msg of messages) {
        openRouterMessages.push({
          role: msg.role,
          content: typeof msg.content === 'string' ? msg.content : msg.content[0]?.text || ''
        });
      }

      const openRouterBody = {
        model: 'mistralai/mistral-7b-instruct:free',
        messages: openRouterMessages,
        max_tokens: 1024,
        temperature: 0.7,
      };

      const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${env.OPENROUTER_API_KEY}`,
          'HTTP-Referer': 'https://mindcoach-app.tiiny.site',
          'X-Title': 'MindCoach AI',
        },
        body: JSON.stringify(openRouterBody),
      });

      const data = await response.json();

      console.log('OpenRouter status:', response.status);
      console.log('OpenRouter data:', JSON.stringify(data).slice(0, 300));

      if (!response.ok) {
        throw new Error(data?.error?.message || 'OpenRouter API error');
      }

      const text = data?.choices?.[0]?.message?.content || 'I am here to help. Please try again.';

      // Return in Anthropic format so the app works without changes
      const result = {
        content: [{ type: 'text', text: text }],
        role: 'assistant',
      };

      return new Response(JSON.stringify(result), {
        status: 200,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      });

    } catch (error) {
      console.error('Worker error:', error.message);
      return new Response(JSON.stringify({
        content: [{ type: 'text', text: 'Error: ' + error.message }],
        role: 'assistant'
      }), {
        status: 200,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      });
    }
  },
};
