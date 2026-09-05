import { createApp, computed, onMounted, ref, watch } from './vendor/vue.esm-browser.prod.js'
import { getMachines, getAllStatus, selectMachine, getDiagnosis, getPublisherStatus, usePolling } from './api.js'

const MachineSelectConfirm = {
  props: { machines: { type: Array, required: true }, modelValue: { type: Number, default: null } },
  emits: ['update:modelValue'],
  setup(props, { emit }) {
    const pending = ref(props.modelValue ?? props.machines[0]?.id ?? 1)
    async function confirm() {
      await selectMachine(pending.value)
      emit('update:modelValue', pending.value)
    }
    return { pending, confirm }
  },
  template: `
    <section class="panel select-panel">
      <div class="select-row">
        <label>機1~5 下拉選單</label>
        <select v-model.number="pending">
          <option v-for="m in machines" :key="m.id" :value="m.id">{{ m.name }}</option>
        </select>
      </div>
      <button class="confirm-btn" @click="confirm">確認鈕</button>
      <p class="hint" v-if="modelValue">目前作用機台：機台{{ modelValue }}</p>
    </section>
  `,
}

const PanelMachineSelectors = {
  props: {
    machines: { type: Array, required: true },
    panel1MachineId: { type: Number, required: true },
    panel2MachineId: { type: Number, required: true },
  },
  emits: ['update:panel1MachineId', 'update:panel2MachineId'],
  setup(props, { emit }) {
    const activeTarget = ref('panel1')
    const nameOf = (id) => props.machines.find((m) => m.id === id)?.name ?? '-'
    function onSelect(event) {
      const id = Number(event.target.value)
      emit(activeTarget.value === 'panel1' ? 'update:panel1MachineId' : 'update:panel2MachineId', id)
    }
    return { activeTarget, nameOf, onSelect }
  },
  template: `
    <section class="panel selector-panel">
      <div class="target-buttons">
        <button class="target-btn" :class="{ active: activeTarget === 'panel1' }" @click="activeTarget = 'panel1'">
          版1要看的機台：{{ nameOf(panel1MachineId) }}
        </button>
        <button class="target-btn" :class="{ active: activeTarget === 'panel2' }" @click="activeTarget = 'panel2'">
          版2要看的機台：{{ nameOf(panel2MachineId) }}
        </button>
      </div>
      <select :value="activeTarget === 'panel1' ? panel1MachineId : panel2MachineId" @change="onSelect">
        <option v-for="m in machines" :key="m.id" :value="m.id">{{ m.name }}</option>
      </select>
      <p class="hint">下拉選單（套用到目前選取的版面）</p>
    </section>
  `,
}

const lightColor = { green: '#3ecf6a', yellow: '#e6b800', red: '#e0473f', gray: '#8a8f98' }

const StatusLights = {
  props: { activeMachineId: { type: Number, default: null } },
  setup() {
    const { data: statuses, error } = usePolling(() => getAllStatus())
    return { statuses, error, lightColor }
  },
  template: `
    <section class="panel status-panel">
      <h2>機1~5狀態</h2>
      <p class="subtitle">簡略(亮燈)</p>
      <p v-if="error" class="error">連線失敗，請確認後端已啟動</p>
      <ul class="status-list">
        <li v-for="m in statuses" :key="m.id" class="status-row" :class="{ active: m.id === activeMachineId }">
          <span class="light" :style="{ backgroundColor: lightColor[m.light] }"></span>
          <span class="name">{{ m.name }}</span>
          <span
            class="fdo-badge"
            :class="{ onboarded: m.fdoOnboarded, stale: m.fdoStale }"
            :title="m.fdoStale ? 'FDO 狀態可能過期' : (m.fdoOnboarded ? 'FDO 已上線' : 'FDO 尚未上線')"
          >FDO {{ m.fdoOnboarded ? '✓' : '✗' }}</span>
          <span class="score">{{ m.score !== null ? m.score.toFixed(1) : '-' }}</span>
        </li>
      </ul>
    </section>
  `,
}

const SCORE_HISTORY_LENGTH = 10 // 每秒 poll 一次，約等於近 10 秒的趨勢

const DiagnosisPanel = {
  props: { machineId: { type: Number, required: true }, machineName: { type: String, required: true } },
  setup(props) {
    const { data: diagnosis, error } = usePolling(() => getDiagnosis(props.machineId))
    const history = ref([])

    watch(
      () => props.machineId,
      () => {
        history.value = []
      },
    )
    watch(diagnosis, (d) => {
      if (d && d.score !== null) {
        history.value = [...history.value, d.score].slice(-SCORE_HISTORY_LENGTH)
      }
    })

    const chartPoints = computed(() => {
      const points = history.value
      if (points.length < 2) return ''
      return points
        .map((score, i) => {
          const x = (i / (points.length - 1)) * 100
          const y = 100 - Math.max(0, Math.min(100, score))
          return `${x},${y}`
        })
        .join(' ')
    })

    return { diagnosis, error, chartPoints }
  },
  template: `
    <section class="panel diagnosis-panel">
      <h2>版1</h2>
      <p class="subtitle">AI 信任分數診斷 · {{ machineName }}</p>
      <p v-if="error" class="error">連線失敗，請確認後端已啟動</p>
      <template v-else-if="diagnosis">
        <p class="status-line">
          狀態：<strong :class="{ blocked: diagnosis.blocked }">{{ diagnosis.status }}</strong>
        </p>
        <div class="details">
          <p class="section-title">詳細資訊</p>
          <ul>
            <li v-for="(line, i) in diagnosis.details" :key="i">{{ line }}</li>
          </ul>
        </div>
        <div class="score-block">
          <p class="section-title">信任分數（近 10 秒趨勢）</p>
          <div class="score-chart-wrap">
            <div class="score-axis"><span>100</span><span>50</span><span>0</span></div>
            <svg class="score-chart" viewBox="0 0 100 100" preserveAspectRatio="none">
              <line class="threshold-line" x1="0" y1="50" x2="100" y2="50" />
              <polyline :points="chartPoints" />
            </svg>
          </div>
          <p>{{ diagnosis.score !== null ? diagnosis.score.toFixed(1) : '尚無資料' }}</p>
        </div>
      </template>
    </section>
  `,
}

const PublisherStatusPanel = {
  props: { machineId: { type: Number, required: true }, machineName: { type: String, required: true } },
  setup(props) {
    const { data: status, error } = usePolling(() => getPublisherStatus(props.machineId))
    const formatTime = (iso) => new Date(iso).toLocaleTimeString()
    return { status, error, formatTime }
  },
  template: `
    <section class="panel messages-panel">
      <h2>版2</h2>
      <p class="subtitle">機台 Publisher 運作狀態 · {{ machineName }}</p>
      <p v-if="error" class="error">連線失敗，請確認後端已啟動</p>
      <template v-else-if="status">
        <p class="status-line">
          狀態：
          <strong :class="{ blocked: status.blocked }">{{ status.blocked ? '已截斷（不正常）' : '正常運作中' }}</strong>
        </p>
        <p v-if="status.blocked" class="hint">剩餘 {{ status.blockedSecondsRemaining.toFixed(1) }} 秒後恢復</p>
        <div class="details">
          <p class="section-title">截斷紀錄</p>
          <ul class="message-list">
            <li v-for="(h, i) in status.history" :key="i" class="message-row error">
              <span class="time">{{ formatTime(h.time) }}</span>
              <span class="content">觸發分數 {{ h.score.toFixed(1) }}</span>
            </li>
            <li v-if="status.history.length === 0" class="empty">尚無截斷紀錄</li>
          </ul>
        </div>
      </template>
    </section>
  `,
}

const App = {
  components: { MachineSelectConfirm, PanelMachineSelectors, StatusLights, DiagnosisPanel, PublisherStatusPanel },
  setup() {
    const machines = ref([])
    const activeMachineId = ref(null)
    const panel1MachineId = ref(1)
    const panel2MachineId = ref(2)

    onMounted(async () => {
      machines.value = await getMachines()
    })

    const nameOf = (id) => machines.value.find((m) => m.id === id)?.name ?? `機台${id}`
    const panel1Name = computed(() => nameOf(panel1MachineId.value))
    const panel2Name = computed(() => nameOf(panel2MachineId.value))

    return { machines, activeMachineId, panel1MachineId, panel2MachineId, panel1Name, panel2Name }
  },
  template: `
    <header class="app-header">
      <h1>機台監控 Demo</h1>
      <p>ROS2 聯邦學習防禦系統 · 即時監控面板</p>
    </header>
    <main class="dashboard" v-if="machines.length">
      <MachineSelectConfirm class="area-select" :machines="machines" v-model="activeMachineId" />
      <PanelMachineSelectors
        class="area-panel-select"
        :machines="machines"
        v-model:panel1MachineId="panel1MachineId"
        v-model:panel2MachineId="panel2MachineId"
      />
      <StatusLights class="area-status" :active-machine-id="activeMachineId" />
      <DiagnosisPanel class="area-panel1" :machine-id="panel1MachineId" :machine-name="panel1Name" />
      <PublisherStatusPanel class="area-panel2" :machine-id="panel2MachineId" :machine-name="panel2Name" />
    </main>
  `,
}

createApp(App).mount('#app')
