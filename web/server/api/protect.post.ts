// Proxy: browser -> Nuxt server -> cloaking engine (Cloud Run).
// Keeps ENGINE_URL server-side and normalizes engine errors for the UI.
export default defineEventHandler(async (event) => {
  const ENGINE = process.env.ENGINE_URL || process.env.NUXT_ENGINE_URL || 'http://localhost:8080'

  const parts = await readMultipartFormData(event)
  const file = parts?.find((p) => p.name === 'image')
  const strength = parts?.find((p) => p.name === 'strength')?.data?.toString() || 'standard'

  if (!file?.data?.length) {
    throw createError({ statusCode: 400, message: 'No image uploaded.' })
  }

  const fd = new FormData()
  fd.append('image', new Blob([file.data], { type: file.type || 'image/jpeg' }), file.filename || 'photo.jpg')
  fd.append('strength', strength)
  fd.append('jpeg_simulate', 'true')

  try {
    return await $fetch(`${ENGINE}/protect`, { method: 'POST', body: fd, timeout: 280000 })
  }
  catch (e: any) {
    const data = e?.response?._data
    throw createError({
      statusCode: e?.response?.status || 502,
      message: data?.message || 'Cloaking engine unreachable. Try again.',
      data,
    })
  }
})
