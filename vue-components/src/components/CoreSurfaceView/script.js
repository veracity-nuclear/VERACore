import { LookupTable } from '../../utils/Colors';

// Lateral face order delivered by Python: WEST, NORTH, EAST, SOUTH.
const WEST = 0;
const NORTH = 1;
const EAST = 2;
const SOUTH = 3;

const CELL = 30;

export default {
  name: 'VeraSurface',
  props: {
    value: {
      type: Array,
      default: () => [],
    },
    selectedI: { type: Number, default: -1 },
    selectedJ: { type: Number, default: -1 },
    colorPreset: { type: String, default: 'jet' },
    colorRange: { type: Array, default: () => [0, 1] },
    activeStyle: {
      type: Object,
      default: () => ({}),
    },
    xLabels: {
      type: Array,
      default: () => ['H', 'G', 'F', 'E', 'D', 'C', 'B', 'A'],
    },
    yLabels: {
      type: Array,
      default: () => ['8', '9', '10', '11', '12', '13', '14', '15'],
    },
    aspectRatio: { type: Number, default: 1 },
    busy: { type: Boolean, default: false },
    dark: { type: Boolean, default: false },
    cellSize: { type: Number, default: 60 },
  },
  watch: {
    selectedI(i) { this.activeI = i; },
    selectedJ(j) { this.activeJ = j; },
    aspectRatio() { this.resize(); },
    value() {
      this.$nextTick(() => this.resize());
    },
    colorPreset() { this.updateLut(); },
    colorRange() { this.updateLut(); },
    dark() { this.updateLut(); },
  },
  data() {
    return {
      activeI: this.selectedI,
      activeJ: this.selectedJ,
      sizeStyle: { width: '100px', height: '100px' },
      scaleStyle: { scale: 1 },
      lutVersion: 0,
    };
  },
  computed: {
    coreWidth() {
      const row = this.value.find((r) => r && r.length);
      return row ? row.length : 0;
    },
  },
  created() {
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.lookupTable = new LookupTable(this.colorPreset, this.colorRange);
    this.updateNanColor();
  },
  mounted() {
    this.measureTarget = this.$el.parentElement || this.$el;
    this.resizeObserver.observe(this.measureTarget);
    this.$nextTick(() => this.resize());
  },
  beforeDestroy() {
    this.resizeObserver.disconnect();
    this.resizeObserver = null;
  },
  methods: {
    updateNanColor() {
      if (this.dark) {
        this.lookupTable.setNanColor(30 / 255, 30 / 255, 30 / 255, 1);
      } else {
        this.lookupTable.setNanColor(1, 1, 1, 1);
      }
    },
    updateLut() {
      this.lookupTable.update(this.colorPreset, this.colorRange);
      this.updateNanColor();
      this.lutVersion++; // invalidate triangle colors
    },
    resize() {
      const target = this.measureTarget || this.$el;
      const { width, height } = target.getBoundingClientRect();
      if (width < 1 || height < 1 || this.coreWidth === 0) {
        return;
      }
      const needed = (this.coreWidth + 1) * CELL;
      const ar = this.aspectRatio || 1;
      const t = Math.min(width / (needed * ar), height / needed);
      this.scaleStyle = { scale: `${ar * t} ${t}` };
      this.sizeStyle = { width: `${needed}px`, height: `${needed}px` };
    },
    hover(i, j) {
      this.activeI = i;
      this.activeJ = j;
    },
    exit() {
      this.activeI = this.selectedI;
      this.activeJ = this.selectedJ;
    },
    hasCell(i, j) {
      const cell = this.value?.[j]?.[i];
      return Array.isArray(cell) && cell.length > 0;
    },
    toStyle(i, j) {
      if (i !== this.activeI || j !== this.activeJ) {
        return {};
      }
      return {
        outline: this.dark ? 'solid 2px white' : 'solid 2px black',
        outlineOffset: '-1px',
        zIndex: 10,
        ...this.activeStyle,
      };
    },
    colorFor(v) {
      // eslint-disable-next-line no-unused-expressions
      this.lutVersion; // establish reactive dependency on LUT changes
      if (v === null || v === undefined || Number.isNaN(v)) {
        return this.dark ? 'rgb(30,30,30)' : 'rgb(255,255,255)';
      }
      const rgb = [];
      this.lookupTable.lookupTable.getColor(v, rgb);
      const r = Math.floor(255 * rgb[0] + 0.5);
      const g = Math.floor(255 * rgb[1] + 0.5);
      const b = Math.floor(255 * rgb[2] + 0.5);
      return `rgb(${r},${g},${b})`;
    },
    cellTriangles(i, j) {
      const cell = this.value?.[j]?.[i];
      if (!Array.isArray(cell) || cell.length === 0) return [];
      const s = this.cellSize;
      const nodeCount = cell.length;
      const side = Math.round(Math.sqrt(nodeCount)); // 1 assembly, 2 nodal
      const step = s / side;

      const tris = [];
      for (let n = 0; n < nodeCount; n++) {
        const faces = cell[n]; // [w, n, e, s]
        const nc = n % side;
        const nr = Math.floor(n / side);
        const nx = nc * step;
        const ny = nr * step;
        const cx = nx + step / 2;
        const cy = ny + step / 2;
        const corners = {
          tl: `${nx.toFixed(2)},${ny.toFixed(2)}`,
          tr: `${(nx + step).toFixed(2)},${ny.toFixed(2)}`,
          bl: `${nx.toFixed(2)},${(ny + step).toFixed(2)}`,
          br: `${(nx + step).toFixed(2)},${(ny + step).toFixed(2)}`,
        };
        const center = `${cx.toFixed(2)},${cy.toFixed(2)}`;
        // NORTH on top (screen up), SOUTH bottom, WEST left, EAST right.
        tris.push({ points: `${corners.tl} ${corners.tr} ${center}`, fill: this.colorFor(faces[NORTH]) });
        tris.push({ points: `${corners.bl} ${corners.br} ${center}`, fill: this.colorFor(faces[SOUTH]) });
        tris.push({ points: `${corners.tl} ${corners.bl} ${center}`, fill: this.colorFor(faces[WEST]) });
        tris.push({ points: `${corners.tr} ${corners.br} ${center}`, fill: this.colorFor(faces[EAST]) });
      }
      return tris;
    },
  },
};