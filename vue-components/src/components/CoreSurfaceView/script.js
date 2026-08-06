import { LookupTable } from '../../utils/Colors';

// Lateral face order delivered by Python: WEST, NORTH, EAST, SOUTH.
const WEST = 0;
const NORTH = 1;
const EAST = 2;
const SOUTH = 3;

const CELL = 30;
const LONG_LABEL = 4;

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
    showLabels: { type: Boolean, default: false },
    decimals: { type: Number, default: 2 },
  },
  watch: {
    selectedI(i) {
      this.activeI = i;
    },
    selectedJ(j) {
      this.activeJ = j;
    },
    aspectRatio() {
      this.resize();
    },
    value() {
      this.$nextTick(() => this.resize());
    },
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
    nodeSide() {
      for (const line of this.value) {
        if (!Array.isArray(line)) continue;
        for (const cell of line) {
          if (Array.isArray(cell) && cell.length) {
            return Math.round(Math.sqrt(cell.length));
          }
        }
      }
      return 1;
    },
    // Shrinks with node count so a nodal cell's four labels still fit.
    labelFontSize() {
      return (this.cellSize / this.nodeSide) * 0.11;
    },
    labelStyle() {
      return {
        paintOrder: 'stroke',
        stroke: 'rgba(255,255,255,0.7)',
        strokeWidth: `${(this.labelFontSize * 0.3).toFixed(2)}px`,
      };
    },
    colorMap() {
      this.lookupTable.update(this.colorPreset, this.colorRange);
      if (this.dark) {
        this.lookupTable.setNanColor(30 / 255, 30 / 255, 30 / 255, 1);
      } else {
        this.lookupTable.setNanColor(1, 1, 1, 1);
      }
      return this.lookupTable;
    },
    // grid[j][i] = triangles for that assembly.
    triangleGrid() {
      const lut = this.colorMap;
      return this.value.map((line) =>
        (Array.isArray(line) ? line : []).map((cell) => this.buildCell(lut, cell))
      );
    },
  },
  created() {
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.lookupTable = new LookupTable(this.colorPreset, this.colorRange);
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
    colorFor(lut, v) {
      if (v === null || v === undefined || Number.isNaN(v)) {
        return this.dark ? 'rgb(30,30,30)' : 'rgb(255,255,255)';
      }
      const rgb = [];
      lut.lookupTable.getColor(v, rgb);
      const r = Math.floor(255 * rgb[0] + 0.5);
      const g = Math.floor(255 * rgb[1] + 0.5);
      const b = Math.floor(255 * rgb[2] + 0.5);
      return `rgb(${r},${g},${b})`;
    },
    formatValue(v) {
      if (v === null || v === undefined || Number.isNaN(v)) {
        return '';
      }
      if (v === 0) {
        return '0';
      }
      const d = this.decimals;
      const abs = Math.abs(v);
      if (abs < 1e-2 || abs >= 1e5) {
        return Number(v)
          .toExponential(d)
          .replace(/\.?0+e/, 'e');
      }
      return Number(v).toFixed(d);
    },
    buildCell(lut, cell) {
      if (!Array.isArray(cell) || cell.length === 0) return [];
      const side = this.nodeSide;
      const step = this.cellSize / side;
      const tris = [];
      for (let n = 0; n < cell.length; n++) {
        const faces = cell[n]; // [w, n, e, s]
        const nx = (n % side) * step;
        const ny = Math.floor(n / side) * step;
        const x0 = nx.toFixed(2);
        const y0 = ny.toFixed(2);
        const x1 = (nx + step).toFixed(2);
        const y1 = (ny + step).toFixed(2);
        const cx = nx + step / 2;
        const cy = ny + step / 2;
        const c = `${cx.toFixed(2)},${cy.toFixed(2)}`;
        // [face, polygon points, label x, label y]
        const geometry = [
          [NORTH, `${x0},${y0} ${x1},${y0} ${c}`, cx, ny + step / 6],
          [SOUTH, `${x0},${y1} ${x1},${y1} ${c}`, cx, ny + (5 * step) / 6],
          [WEST, `${x0},${y0} ${x0},${y1} ${c}`, nx + step / 6, cy],
          [EAST, `${x1},${y0} ${x1},${y1} ${c}`, nx + (5 * step) / 6, cy],
        ];
        for (const [face, points, x, y] of geometry) {
          const text = this.showLabels ? this.formatValue(faces[face]) : '';
          tris.push({
            points,
            x,
            y,
            text,
            fill: this.colorFor(lut, faces[face]),
            size: text.length > LONG_LABEL ? this.labelFontSize * 0.75 : this.labelFontSize,
          });
        }
      }
      return tris;
    },
  },
};
