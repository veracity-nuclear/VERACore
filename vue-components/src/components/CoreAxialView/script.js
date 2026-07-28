export default {
  name: 'VeraCoreAxial',
  props: {
    value: { type: Array, default: () => [] },
    xRange: { type: Array, default: () => [0, 1] },
    labels: { type: Array, default: () => [] },
    selectedI: { type: Number, default: -1 },
    selectedJ: { type: Number, default: -1 },
    xLabels: { type: Array, default: () => ['H', 'G', 'F', 'E', 'D', 'C', 'B', 'A'] },
    yLabels: { type: Array, default: () => ['8', '9', '10', '11', '12', '13', '14', '15'] },
    activeStyle: {
      type: Object,
      default: () => ({ outline: 'solid 1px black', zIndex: 10 }),
    },
    aspectRatio: { type: Number, default: 1 },
    busy: { type: Boolean, default: false },
    cellSize: { type: Number, default: 48 },
  },
  watch: {
    selectedI(i) { this.activeI = i; },
    selectedJ(j) { this.activeJ = j; },
    aspectRatio() { this.resize(); },
  },
  data() {
    return {
      activeI: this.selectedI,
      activeJ: this.selectedJ,
      sizeStyle: { width: '100px', height: '100px' },
      scaleStyle: { scale: 1 },
    };
  },
  computed: {
    coreWidth() {
      const row = this.value.find((r) => r && r.length);
      return row ? row.length : 0;
    },
    gridLines() {
        const s = this.cellSize;
        const divs = 4;
        const step = s / divs;
        const lines = [];
        for (let k = 1; k < divs; k++) {
        const p = (k * step).toFixed(2);
        lines.push({ x1: p, y1: 0, x2: p, y2: s });
        lines.push({ x1: 0, y1: p, x2: s, y2: p });
        }
        return lines;
    },
  },
  created() {
    this.resizeObserver = new ResizeObserver(() => this.resize());
  },
  mounted() {
    this.resizeObserver.observe(this.$el);
  },
  beforeDestroy() {
    this.resizeObserver.disconnect();
    this.resizeObserver = null;
  },
  methods: {
    resize() {
      const { width, height } = this.$el.getBoundingClientRect();
      const needed = (this.coreWidth + 1) * 32;
      const ar = this.aspectRatio || 1;
      const t = Math.min(width / (needed * ar), height / needed);
      this.scaleStyle = { scale: `${ar * t} ${t}` };
      this.sizeStyle = { width: `${needed + 10}px`, height: `${needed + 10}px` };
    },
    hover(i, j) {
      this.activeI = i;
      this.activeJ = j;
    },
    exit() {
      this.activeI = this.selectedI;
      this.activeJ = this.selectedJ;
    },
    toStyle(i, j) {
      return (i === this.activeI && j === this.activeJ) ? this.activeStyle : {};
    },
    hasCell(i, j) {
      const cell = this.value?.[j]?.[i];
      return Array.isArray(cell) && cell.length > 1;
    },
    toPoints(i, j) {
      const profile = this.value?.[j]?.[i];
      if (!Array.isArray(profile) || profile.length < 2) return '';
      const [xMin, xMax] = this.xRange;
      const span = (xMax > xMin) ? (xMax - xMin) : 1;
      const s = this.cellSize;
      const n = profile.length;
      let out = '';
      for (let k = 0; k < n; k++) {
        const xs = ((profile[k] - xMin) / span) * (s - 1);
        const ys = (1 - k / (n - 1)) * (s - 1);
        out += `${xs.toFixed(2)},${ys.toFixed(2)} `;
      }
      return out.trim();
    },
    toLabel(i, j) {
      const v = this.labels?.[j]?.[i];
      if (v === undefined || v === null || Number.isNaN(v)) return '';
      return Number(v).toFixed(2);
    },
  },
};