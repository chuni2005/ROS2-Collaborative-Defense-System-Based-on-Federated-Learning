import { onMounted, onUnmounted, ref } from './vendor/vue.esm-browser.prod.js'

async function request(path, options) {
  const res = await fetch(path, options)
  if (!res.ok) throw new Error(`API ${path} failed: ${res.status}`)
  return res.json()
}

export const getMachines = () => request('/api/machines')
export const getAllStatus = () => request('/api/machines/status')
export const selectMachine = (machineId) =>
  request('/api/machines/select', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ machineId }),
  })
export const getDiagnosis = (machineId) => request(`/api/machines/${machineId}/diagnosis`)
export const getMessages = (machineId) => request(`/api/machines/${machineId}/messages`)

export function usePolling(fetchFn, { intervalMs = 1000, immediate = true } = {}) {
  const data = ref(null)
  const error = ref(null)
  let timer = null

  async function tick() {
    try {
      data.value = await fetchFn()
      error.value = null
    } catch (err) {
      error.value = err
    }
  }
  function start() {
    stop()
    if (immediate) tick()
    timer = setInterval(tick, intervalMs)
  }
  function stop() {
    if (timer) {
      clearInterval(timer)
      timer = null
    }
  }
  onMounted(start)
  onUnmounted(stop)
  return { data, error, refresh: tick }
}
