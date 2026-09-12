import { LookupTable } from '../../utils/Colors';
import { toImageURL } from '../../utils/ImageGenerator';

function indexAt(edges, pos) {
  for (let k = edges.length - 2; k >= 0; k -= 1) {
    if (pos >= edges[k]) {
      return k;
    }
  }
  return 0;
}

function edgesFrom(sizes, scale) {
  const edges = [0];
  let acc = 0;
  for (let k = 0; k < sizes.length; k += 1) {
    acc += sizes[k] * scale;
    edges.push(Math.round(acc));
  }
  return edges;
}

function centersOf(edges) {
  const centers = [];
  for (let k = 0; k < edges.length - 1; k += 1) {
    centers.push((edges[k] + edges[k + 1]) / 2);
  }
  return centers;
}

function decimate(centers, gap) {
  const n = centers.length;
  if (n < 2) {
    return n ? [0] : [];
  }
  const keep = [0];
  for (let k = 1; k < n - 1; k += 1) {
    if (centers[k] - centers[keep[keep.length - 1]] >= gap) {
      keep.push(k);
    }
  }
  while (keep.length > 1 && centers[n - 1] - centers[keep[keep.length - 1]] < gap) {
    keep.pop();
  }
  if (centers[n - 1] - centers[keep[keep.length - 1]] >= gap) {
    keep.push(n - 1);
  }
  return keep;
}

function niceStep(raw) {
  if (!(raw > 0)) {
    return 1;
  }
  const mag = 10 ** Math.floor(Math.log10(raw));
  const norm = raw / mag;
  let nice = 10;
  if (norm <= 1) nice = 1;
  else if (norm <= 2) nice = 2;
  else if (norm <= 5) nice = 5;
  return nice * mag;
}

function formatTick(value, step) {
  return step >= 1 ? String(Math.round(value)) : value.toFixed(1);
}

export default {
  name: 'VeraAxial',
  props: {
    value: {
      type: Array,
      default: () => [[[], [], []], [[], []], [[]]],
    },
    selectedI: {
      type: Number,
      default: -1,
    },
    selectedJ: {
      type: Number,
      default: -1,
    },
    colorPreset: {
      type: String,
      default: 'erdc_rainbow_bright',
    },
    colorRange: {
      type: Array,
      default: () => [0, 1],
    },
    activeStyle: {
      type: Object,
      default: () => ({
        outline: 'solid 1px black',
        zIndex: 10,
      }),
    },
    xLabels: {
      type: Array,
      default: () => ['H', 'G', 'F', 'E', 'D', 'C', 'B', 'A'],
    },
    yLabels: {
      type: Array,
      default: () => ['8', '9', '10', '11', '12', '13', '14', '15'],
    },
    yBounds: {
      type: Array,
      default: () => [],
    },
    yUnit: {
      type: String,
      default: 'cm',
    },
    xScale: {
      type: Number,
      default: 2,
    },
    yScale: {
      type: Number,
      default: 3,
    },
    xSizes: {
      type: Array,
      default: () => [],
    },
    ySizes: {
      type: Array,
      default: () => [],
    },
    fontSize: {
      type: Number,
      default: 11,
    },
    labelPadding: {
      type: Number,
      default: 6,
    },
    maxCellSize: {
      type: Number,
      default: 64,
    },
    busy: {
      type: Boolean,
      default: false,
    },
    dark: {
      type: Boolean,
      default: false,
    },
  },
  watch: {
    selectedI(i) {
      this.activeI = i;
    },
    selectedJ(j) {
      this.activeJ = j;
    },
  },
  data() {
    return {
      activeI: this.selectedI,
      activeJ: this.selectedJ,
      boxWidth: 0,
      boxHeight: 0,
      fontFamily: 'sans-serif',
      measureNonce: 0,
      hover: null,
    };
  },
  computed: {
    rowCount() {
      return Math.min(this.value.length, this.ySizes.length);
    },
    columnCount() {
      return this.xSizes.length;
    },
    // NaN cells are drawn just off the panel background so an absent assembly
    // reads as an empty cell rather than as a hole in the grid.
    nanColor() {
      const level = this.dark ? 45 / 255 : 0.93;
      return [level, level, level, 1];
    },
    colorMap() {
      const [r, g, b, a] = this.nanColor;
      this.lookupTable.setNanColor(r, g, b, a);
      return this.lookupTable.update(this.colorPreset, this.colorRange);
    },
    rowImages() {
      const lut = this.colorMap;
      const urls = [];
      for (let j = 0; j < this.value.length; j += 1) {
        const pixels = this.value[j].flat();
        urls.push(pixels.length ? toImageURL(lut, pixels, pixels.length, 1, this.xScale, 1) : null);
      }
      return urls;
    },
    renderedRows() {
      const rows = [];
      const edges = this.yEdges;
      for (let j = 0; j < this.rowCount; j += 1) {
        if (this.rowImages[j] && edges[j + 1] != null) {
          rows.push({ j, url: this.rowImages[j] });
        }
      }
      return rows;
    },
    lineHeight() {
      return Math.ceil(this.fontSize * 1.4);
    },
    widestXLabel() {
      let width = 0;
      for (let i = 0; i < this.columnCount; i += 1) {
        width = Math.max(width, this.textWidth(this.xLabels[i]));
      }
      return Math.ceil(width);
    },
    xSlack() {
      return Math.ceil(this.widestXLabel / 2);
    },
    ySlack() {
      return Math.ceil(this.lineHeight / 2);
    },
    gutterWidth() {
      let width = 0;
      for (let j = 0; j < this.yLabels.length; j += 1) {
        width = Math.max(width, this.textWidth(this.yLabels[j]));
      }
      for (let t = 0; t < this.yTicks.length; t += 1) {
        width = Math.max(width, this.textWidth(this.yTicks[t].text));
      }
      return width ? Math.ceil(width) + this.labelPadding : 0;
    },
    headerHeight() {
      return this.widestXLabel ? this.lineHeight : 0;
    },
    xLabelGap() {
      return this.widestXLabel + this.labelPadding;
    },
    yLabelGap() {
      return this.lineHeight + 4;
    },
    availWidth() {
      return Math.max(0, this.boxWidth - this.gutterWidth - this.xSlack * 2);
    },
    availHeight() {
      return Math.max(0, this.boxHeight - this.headerHeight - this.ySlack * 2);
    },
    unitWidth() {
      return this.xSizes.reduce((sum, size) => sum + size * this.xScale, 0);
    },
    unitHeight() {
      return this.ySizes.reduce((sum, size) => sum + size * this.yScale, 0);
    },
    fit() {
      if (!this.unitWidth || !this.unitHeight) {
        return 0;
      }
      const f = Math.min(
        this.availWidth / this.unitWidth,
        this.availHeight / this.unitHeight,
        this.maxCellSize / (Math.max(...this.xSizes) * this.xScale),
        this.maxCellSize / (Math.max(...this.ySizes) * this.yScale),
      );
      return Number.isFinite(f) ? f : 0;
    },
    xEdges() { return edgesFrom(this.xSizes, this.xScale * this.fit); },
    yEdges() { return edgesFrom(this.ySizes, this.yScale * this.fit); },
    dataWidth() {
      return this.xEdges[this.xEdges.length - 1];
    },
    dataHeight() {
      return this.yEdges[this.yEdges.length - 1];
    },
    useTicks() {
      return this.yBounds.length >= 2 && this.yBounds.length === this.yEdges.length;
    },
    visibleColumns() {
      return decimate(centersOf(this.xEdges), this.xLabelGap);
    },
    visibleRows() {
      if (this.useTicks) {
        return [];
      }
      return decimate(centersOf(this.yEdges).slice(0, this.rowCount), this.yLabelGap);
    },
    yTicks() {
      const b = this.yBounds;
      if (!this.useTicks || !this.dataHeight) {
        return [];
      }
      const hi = Math.max(b[0], b[b.length - 1]);
      const lo = Math.min(b[0], b[b.length - 1]);
      const target = Math.max(2, Math.floor(this.dataHeight / (this.lineHeight * 1.8)));
      const step = niceStep((hi - lo) / target);
      const ticks = [];
      const start = Math.ceil(lo / step) * step;
      for (let v = start; v <= hi + 1e-9; v += step) {
        ticks.push({ value: v, text: formatTick(v, step), y: this.pixelForCm(v) });
      }
      return ticks;
    },
    frameLeft() {
      const total = this.gutterWidth + this.dataWidth;
      const centered = Math.round((this.boxWidth - total) / 2);
      const maxLeft = this.boxWidth - total - this.xSlack;
      return Math.max(0, Math.min(centered, maxLeft));
    },
    frameTop() {
      const total = this.headerHeight + this.dataHeight;
      const centered = Math.round((this.boxHeight - total) / 2);
      const maxTop = this.boxHeight - total - this.ySlack;
      return Math.max(this.ySlack, Math.min(centered, maxTop));
    },
    frameStyle() {
      return {
        position: 'absolute',
        left: `${this.frameLeft}px`,
        top: `${this.frameTop}px`,
        width: `${this.gutterWidth + this.dataWidth}px`,
        height: `${this.headerHeight + this.dataHeight}px`,
        fontSize: `${this.fontSize}px`,
        lineHeight: `${this.lineHeight}px`,
      };
    },
    dataStyle() {
      return {
        position: 'absolute',
        left: `${this.gutterWidth}px`,
        top: `${this.headerHeight}px`,
        width: `${this.dataWidth}px`,
        height: `${this.dataHeight}px`,
      };
    },
    highlightStyle() {
      const i = this.activeI;
      const j = this.activeJ;
      if (i < 0 || j < 0 || i >= this.columnCount || j >= this.rowCount) {
        return { display: 'none' };
      }
      return {
        position: 'absolute',
        left: `${this.xEdges[i]}px`,
        top: `${this.yEdges[j]}px`,
        width: `${this.xEdges[i + 1] - this.xEdges[i]}px`,
        height: `${this.yEdges[j + 1] - this.yEdges[j]}px`,
        pointerEvents: 'none',
        ...this.activeStyle,
      };
    },
    tooltipStyle() {
      if (!this.hover) {
        return { display: 'none' };
      }
      const x = this.frameLeft + this.gutterWidth + this.hover.x;
      const y = this.frameTop + this.headerHeight + this.hover.y;
      const flip = x > this.boxWidth * 0.6;
      return {
        position: 'absolute',
        left: flip ? 'auto' : `${x + 12}px`,
        right: flip ? `${this.boxWidth - x + 12}px` : 'auto',
        top: `${Math.max(0, y - 10)}px`,
        pointerEvents: 'none',
        whiteSpace: 'nowrap',
        zIndex: 20,
      };
    },
    tooltipText() {
      if (!this.hover) {
        return '';
      }
      const label = this.yLabels[this.hover.j];
      return label == null ? '' : `${label} ${this.yUnit}`;
    },
  },
  created() {
    this.lookupTable = new LookupTable(this.colorPreset, this.colorRange);
    this.measureContext = null;
    this.measureWarned = false;
    this.resizeObserver = new ResizeObserver(() => this.resize());
  },
  mounted() {
    this.fontFamily = window.getComputedStyle(this.$el).fontFamily || 'sans-serif';
    this.resizeObserver.observe(this.$el);
    if (document.fonts && document.fonts.ready) {
      document.fonts.ready.then(() => {
        this.fontFamily = window.getComputedStyle(this.$el).fontFamily || 'sans-serif';
        this.measureNonce += 1;
        this.resize();
      });
    }
  },
  beforeDestroy() {
    this.resizeObserver.disconnect();
    this.resizeObserver = null;
  },
  methods: {
    resize() {
      const { width, height } = this.$el.getBoundingClientRect();
      this.boxWidth = width;
      this.boxHeight = height;
    },
    ctx() {
      if (!this.measureContext) {
        try {
          this.measureContext = document.createElement('canvas').getContext('2d');
        } catch (e) {
          this.measureContext = null;
        }
      }
      return this.measureContext;
    },
    textWidth(text) {
      this.measureNonce;
      const value = text == null ? '' : String(text);
      if (!value) {
        return 0;
      }
      const context = this.ctx();
      if (!context) {
        if (!this.measureWarned) {
          this.measureWarned = true;
          // eslint-disable-next-line no-console
          console.warn('VeraAxial: canvas measure unavailable, using width estimate');
        }
        return value.length * this.fontSize * 0.6;
      }
      context.font = `${this.fontSize}px ${this.fontFamily}`;
      return context.measureText(value).width;
    },
    pixelForCm(v) {
      const b = this.yBounds;
      const e = this.yEdges;
      if (v >= b[0]) return e[0];
      if (v <= b[b.length - 1]) return e[e.length - 1];
      for (let k = 0; k < b.length - 1; k += 1) {
        if (v <= b[k] && v >= b[k + 1]) {
          const span = b[k] - b[k + 1] || 1;
          const t = (b[k] - v) / span;
          return e[k] + t * (e[k + 1] - e[k]);
        }
      }
      return e[e.length - 1];
    },
    xLabelStyle(i) {
      const center = (this.xEdges[i] + this.xEdges[i + 1]) / 2;
      return {
        position: 'absolute',
        left: `${this.gutterWidth + center}px`,
        top: 0,
        height: `${this.headerHeight}px`,
        display: 'flex',
        alignItems: 'center',
        transform: 'translateX(-50%)',
      };
    },
    yLabelStyle(j) {
      const center = (this.yEdges[j] + this.yEdges[j + 1]) / 2;
      return {
        position: 'absolute',
        right: `${this.dataWidth + this.labelPadding}px`,
        top: `${this.headerHeight + center}px`,
        transform: 'translateY(-50%)',
      };
    },
    tickLabelStyle(y) {
      return {
        position: 'absolute',
        right: `${this.dataWidth + this.labelPadding}px`,
        top: `${this.headerHeight + y}px`,
        transform: 'translateY(-50%)',
      };
    },
    tickMarkStyle(y) {
      return {
        position: 'absolute',
        left: `${this.gutterWidth - 3}px`,
        top: `${this.headerHeight + Math.round(y)}px`,
        width: '3px',
        height: '1px',
      };
    },
    rowStyle(j) {
      return {
        position: 'absolute',
        left: 0,
        top: `${this.yEdges[j]}px`,
        width: '100%',
        height: `${this.yEdges[j + 1] - this.yEdges[j]}px`,
        display: 'block',
        pointerEvents: 'none',
      };
    },
    locate(event) {
      const rect = event.currentTarget.getBoundingClientRect();
      const lx = event.clientX - rect.left;
      const ly = event.clientY - rect.top;
      return { lx, ly, i: indexAt(this.xEdges, lx), j: indexAt(this.yEdges, ly) };
    },
    onClick(event) {
      const { i, j } = this.locate(event);
      this.$emit('click', { i, j });
    },
    onHover(event) {
      const { lx, ly, i, j } = this.locate(event);
      this.activeI = i;
      this.activeJ = j;
      this.hover = { x: lx, y: ly, j };
    },
    exit() {
      this.activeI = this.selectedI;
      this.activeJ = this.selectedJ;
      this.hover = null;
    },
  },
};