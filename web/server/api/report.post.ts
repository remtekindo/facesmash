// Proxy: browser -> Nuxt server -> engine /report (Gemini, mock fallback).
export default defineEventHandler(async (event) => {
  const ENGINE = process.env.ENGINE_URL || process.env.NUXT_ENGINE_URL || 'http://localhost:8080'
  const body = await readBody(event)

  try {
    return await $fetch(`${ENGINE}/report`, {
      method: 'POST',
      body: {
        metrics: body.metrics,
        meta: { strength: body.meta?.strength || 'standard', processing_time_ms: body.meta?.processing_time_ms || 0, faces_protected: body.meta?.faces_protected || 1 },
        verification: body.verification || {},
        lang: body.lang || 'en',
      },
      timeout: 60000,
    })
  }
  catch {
    throw createError({ statusCode: 502, message: 'Report service unreachable. Try again.' })
  }
})
