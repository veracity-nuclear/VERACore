import { LookupTable } from '../../utils/Colors';

const GUTTER = 30; // label gutter
const CELL_GAP = 6;
const PLOT_PADDING = 3; // border to data
const GRID_DIVISIONS = 2;
const LABEL_FONT = 11;
const POINT_RADIUS = 0.7;
const BORDER_WIDTH = 1.5;
const SERIES_WIDTH = 0.7;
const BRACKET_OFFSET = 2;
const BRACKET_WIDTH = 2.5;
const BRACKET_FRACTION = 0.3;
const HALO_EXTRA = 2;

const THEMES = {
  light: {
    surface: '#ffffff',
    surfaceHover: '#e6edf5',
    grid: '#dddddd',
    stroke: '#000000',
    label: '#000000',
    empty: '#e0e0e0',
    highlight: '#000000',
    halo: '#ffffff',
    reference: '#c4c4c4',
  },
  dark: {
    surface: '#1e1e1e',
    surfaceHover: '#30373f',
    grid: '#3a3a3a',
    stroke: '#e0e0e0',
    label: '#e0e0e0',
    empty: '#4a4a4a',
    highlight: '#ffffff',
    halo: '#000000',
    reference: '#5a5a5a',
  },
};

export default {
  name: 'VeraCoreAxial',
  props: {
    value: { type: Array, default: () => [] },
    mesh: { type: Object, default: () => ({ kind: 'line', y: [] }) },
    xRange: { type: Array, default: () => [0, 1] },
    yRange: { type: Array, default: () => [0, 1] },
    labels: { type: Array, default: () => [] },
    selectedI: { type: Number, default: -1 },
    selectedJ: { type: Number, default: -1 },
    xLabels: { type: Array, default: () => ['H', 'G', 'F', 'E', 'D', 'C', 'B', 'A'] },
    yLabels: { type: Array, default: () => ['8', '9', '10', '11', '12', '13', '14', '15'] },
    aspectRatio: { type: Number, default: 1 },
    busy: { type: Boolean, default: false },
    cellSize: { type: Number, default: 48 },
    colorPreset: { type: String, default: 'jet' },
    colorRange: { type: Array, default: () => [0, 1] },
    dark: { type: Boolean, default: false },
    highlightStroke: { type: String, default: '' },
  },
  data() {
    return {
      hoverI: -1,
      hoverJ: -1,
    };
  },
  computed: {
    activeI() {
      return this.hoverI >= 0 ? this.hoverI : this.selectedI;
    },
    activeJ() {
      return this.hoverJ >= 0 ? this.hoverJ : this.selectedJ;
    },
    theme() {
      return this.dark ? THEMES.dark : THEMES.light;
    },
    gutter() {
      return GUTTER;
    },
    labelFont() {
      return LABEL_FONT;
    },
    pointDiameter() {
      return POINT_RADIUS * 2;
    },
    borderWidth() {
      return BORDER_WIDTH;
    },
    seriesWidth() {
      return SERIES_WIDTH;
    },
    bracketWidth() {
      return BRACKET_WIDTH;
    },
    haloWidth() {
      return BRACKET_WIDTH + HALO_EXTRA;
    },
    selectionStroke() {
      return this.highlightStroke || this.theme.highlight;
    },
    cellW() {
      return this.cellSize * (this.aspectRatio || 1);
    },
    cellH() {
      return this.cellSize;
    },
    box() {
      const margin = CELL_GAP / 2;
      return {
        x: margin,
        y: margin,
        width: this.cellW - CELL_GAP,
        height: this.cellH - CELL_GAP,
      };
    },
    rowCount() {
      return this.value.length;
    },
    coreWidth() {
      const row = this.value.find((r) => r && r.length);
      return row ? row.length : 0;
    },
    viewBox() {
      const w = GUTTER + this.coreWidth * this.cellW;
      const h = GUTTER + this.rowCount * this.cellH;
      return `0 0 ${w} ${h}`;
    },
    plot() {
      const { x, y, width, height } = this.box;
      return {
        left: x + PLOT_PADDING,
        top: y + PLOT_PADDING,
        width: width - 2 * PLOT_PADDING,
        height: height - 2 * PLOT_PADDING,
      };
    },
    gridLines() {
      const { x, y, width, height } = this.box;
      const lines = [];
      for (let k = 1; k < GRID_DIVISIONS; k++) {
        const gy = (y + (k * height) / GRID_DIVISIONS).toFixed(2);
        lines.push({ x1: x, y1: gy, x2: x + width, y2: gy });
      }
      return lines;
    },
    referenceLine() {
      const [lo, hi] = this.xRange;
      if (!Number.isFinite(lo) || !Number.isFinite(hi)) {
        return null;
      }
      const x = this.toX((lo + hi) / 2).toFixed(2);
      return { x1: x, y1: this.plot.top, x2: x, y2: this.plot.top + this.plot.height };
    },
    cells() {
      const px = (v) => this.toX(v).toFixed(2);
      const py = (v) => this.toY(v).toFixed(2);

      const y = (this.mesh && this.mesh.y) || [];
      const kind = (this.mesh && this.mesh.kind) || 'line';

      return this.value.map((row, j) => row.map((cell, i) => {
        const base = {
          i,
          j,
          tx: GUTTER + i * this.cellW,
          ty: GUTTER + j * this.cellH,
          label: this.formatLabel(i, j),
          border: this.theme.empty,
          shape: null,
        };
        if (!cell || !cell.x || !cell.x.length || !y.length) {
          return base;
        }
        base.shape = this.toShape(cell.x, y, kind, px, py);
        base.border = this.colorFor(cell.mean);
        return base;
      }));
    },
    selectionPath() {
      const cell = this.cellAt(this.selectedI, this.selectedJ);
      if (!cell || !cell.shape) {
        return '';
      }
      const x = cell.tx + this.box.x - BRACKET_OFFSET;
      const y = cell.ty + this.box.y - BRACKET_OFFSET;
      const w = this.box.width + 2 * BRACKET_OFFSET;
      const h = this.box.height + 2 * BRACKET_OFFSET;
      const a = Math.min(w, h) * BRACKET_FRACTION;
      return [
        `M${x},${y + a}L${x},${y}L${x + a},${y}`,
        `M${x + w - a},${y}L${x + w},${y}L${x + w},${y + a}`,
        `M${x + w},${y + h - a}L${x + w},${y + h}L${x + w - a},${y + h}`,
        `M${x + a},${y + h}L${x},${y + h}L${x},${y + h - a}`,
      ].join('');
    },
    colorMap() {
      return this.lookupTable.update(this.colorPreset, this.colorRange);
    },
  },
  created() {
    this.lookupTable = new LookupTable(this.colorPreset, this.colorRange);
  },
  methods: {
    toX(value) {
      const [lo, hi] = this.xRange;
      const span = hi > lo ? hi - lo : 1;
      return this.plot.left + ((value - lo) / span) * this.plot.width;
    },
    toY(value) {
      const [lo, hi] = this.yRange;
      const span = hi > lo ? hi - lo : 1;
      return this.plot.top + (1 - (value - lo) / span) * this.plot.height;
    },
    cellAt(i, j) {
      const row = this.cells[j];
      return row ? row[i] : undefined;
    },
    isHovered(cell) {
      return cell.i === this.hoverI && cell.j === this.hoverJ;
    },
    colorFor(mean) {
      if (mean == null || !Number.isFinite(mean)) {
        return this.theme.empty;
      }
      const [r, g, b] = this.colorMap.getRGB(mean);
      return `rgb(${Math.round(r * 255)}, ${Math.round(g * 255)}, ${Math.round(b * 255)})`;
    },
    toShape(x, y, kind, px, py) {
      if (x.length !== y.length) {
        if (!this._zipWarned) {
          this._zipWarned = true;
          console.warn('VeraCoreAxial: cell x length', x.length, 'does not match mesh y length', y.length);
        }
        return null;
      }
      const color = this.theme.stroke;

      if (kind === 'points') {
        let d = '';
        for (let k = 0; k < x.length; k++) {
          if (x[k] != null && y[k] != null) {
            d += `M${px(x[k])},${py(y[k])}h0`;
          }
        }
        return d ? { kind: 'points', d, color } : null;
      }

      let d = '';
      let drawing = false;
      for (let k = 0; k < x.length; k++) {
        if (x[k] == null || y[k] == null) {
          drawing = false;
          continue;
        }
        d += `${drawing ? 'L' : 'M'}${px(x[k])},${py(y[k])}`;
        drawing = true;
      }
      return d ? { kind: 'path', d, color } : null;
    },
    formatLabel(i, j) {
      const v = this.labels && this.labels[j] && this.labels[j][i];
      if (v === undefined || v === null || Number.isNaN(v)) {
        return '';
      }
      return Number(v).toFixed(2);
    },
    hover(cell) {
      if (!cell.shape) {
        this.exit();
        return;
      }
      this.hoverI = cell.i;
      this.hoverJ = cell.j;
    },
    exit() {
      this.hoverI = -1;
      this.hoverJ = -1;
    },
    select(cell) {
      if (cell.shape) {
        this.$emit('click', { i: cell.i, j: cell.j });
      }
    },
    columnX(i) {
      return GUTTER + (i + 0.5) * this.cellW;
    },
    rowY(j) {
      return GUTTER + (j + 0.5) * this.cellH;
    },
  },
};