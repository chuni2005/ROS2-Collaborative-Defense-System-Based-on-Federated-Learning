import { createApp, computed, onMounted, ref } from './vendor/vue.esm-browser.prod.js'
import { getMachines, getAllStatus, selectMachine, getDiagnosis, getMessages, usePolling } from './api.js'

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
        </li>
      </ul>
    </section>
  `,
}

const DiagnosisPanel = {
  props: { machineId: { type: Number, required: true }, machineName: { type: String, required: true } },
  setup(props) {
    const { data: diagnosis, error } = usePolling(() => getDiagnosis(props.machineId))
    return { diagnosis, error }
  },
  template: `
    <section class="panel diagnosis-panel">
      <h2>版1</h2>
      <p class="subtitle">後端診斷的機台狀態 · {{ machineName }}</p>
      <p v-if="error" class="error">連線失敗，請確認後端已啟動</p>
      <template v-else-if="diagnosis">
        <p class="status-line">狀態：<strong>{{ diagnosis.status }}</strong></p>
        <div class="details">
          <p class="section-title">詳細資訊</p>
          <ul>
            <li v-for="(line, i) in diagnosis.details" :key="i">{{ line }}</li>
          </ul>
        </div>
        <div class="score-block">
          <p class="section-title">分數與持續時間之類的</p>
          <p>信心分數：{{ diagnosis.score.toFixed(1) }}</p>
          <p>持續時間：{{ diagnosis.durationSeconds }} 秒</p>
        </div>
      </template>
    </section>
  `,
}

const MessagesPanel = {
  props: { machineId: { type: Number, required: true }, machineName: { type: String, required: true } },
  setup(props) {
    const { data: messages, error } = usePolling(() => getMessages(props.machineId))
    const formatTime = (iso) => new Date(iso).toLocaleTimeString()
    return { messages, error, formatTime }
  },
  template: `
    <section class="panel messages-panel">
      <h2>版2</h2>
      <p class="subtitle">機台送出的訊息 · {{ machineName }}</p>
      <p v-if="error" class="error">連線失敗，請確認後端已啟動</p>
      <ul v-else class="message-list">
        <li v-for="(msg, i) in messages" :key="i" class="message-row" :class="msg.level">
          <span class="time">{{ formatTime(msg.timestamp) }}</span>
          <span class="content">{{ msg.content }}</span>
        </li>
        <li v-if="messages && messages.length === 0" class="empty">尚無訊息</li>
      </ul>
    </section>
  `,
}

const App = {
  components: { MachineSelectConfirm, PanelMachineSelectors, StatusLights, DiagnosisPanel, MessagesPanel },
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
      <MessagesPanel class="area-panel2" :machine-id="panel2MachineId" :machine-name="panel2Name" />
    </main>
  `,
}

createApp(App).mount('#app')
