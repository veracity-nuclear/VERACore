import { LookupTable } from '../../utils/Colors';

const CELL = 60;
const NAN_DARK = 'rgb(30, 30, 30)';
const NAN_LIGHT = 'rgb(255, 255, 255)';

function css(rgb) {
  const [r, g, b] = rgb;
  return `rgb(${Math.round(r * 255)}, ${Math.round(g * 255)}, ${Math.round(b * 255)})`;
}

function readableOn(rgb) {
  const [r, g, b] = rgb;
  return 0.299 * r + 0.587 * g + 0.114 * b > 0.6 ? '#000000' : '#ffffff';
}

export default {
  name: 'VeraCipsCore',
  props: {
    // [{ index, label, name, units, values: (number|null)[][] }]
    metrics: { type: Array, default: () => [] },
    colorRanges: { type: Array, default: () => [] },
    colorPreset: { type: String, default: 'jet' },
    colorMode: { type: String, default: 'primary' }, // 'primary' | 'banded'
    primaryIndex: { type: Number, default: 0 },
    selectedI: { type: Number, default: -1 },
    selectedJ: { type: Number, default: -1 },
    activeStyle: { type: Object, default: () => ({}) },
    xLabels: { type: Array, default: () => [] },
    yLabels: { type: Array, default: () => [] },
    coreCols: { type: Number, default: 0 },
    aspectRatio: { type: Number, default: 1 },
    decimals: { type: Number, default: 2 },
    dark: { type: Boolean, default: false },
    busy: { type: Boolean, default: false },
  },
  data() {
    return {
      activeI: this.selectedI,
      activeJ: this.selectedJ,
      sizeStyle: { width: '100px', height: '100px' },
      scaleStyle: { scale: 1 },
      cell: CELL,
    };
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
    metrics() {
      this.$nextTick(() => this.resize());
    },
    coreCols() {
      this.$nextTick(() => this.resize());
    },
  },
  computed: {
    coreWidth() {
      return this.coreCols || this.grid.reduce((m, row) => Math.max(m, row.length), 0);
    },
    grid() {
      return this.metrics[0] ? this.metrics[0].values : [];
    },
    bandCount() {
      return Math.max(this.metrics.length, 1);
    },
    bandHeight() {
      return CELL / this.bandCount;
    },
    fontSize() {
      return Math.max(8, Math.round(this.bandHeight * 0.45));
    },
    nanColor() {
      return this.dark ? NAN_DARK : NAN_LIGHT;
    },
    textOnNan() {
      return this.dark ? '#ffffff' : '#000000';
    },
    // One lookup table per metric; rebuilt whenever metrics or ranges change.
    luts() {
      return this.metrics.map((metric, k) => {
        const range = this.colorRanges[k] || [0, 1];
        const lut = new LookupTable(this.colorPreset, range);
        lut.update(this.colorPreset, range);
        return lut;
      });
    },
  },
  created() {
    this.resizeObserver = new ResizeObserver(() => this.resize());
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
    valueAt(k, i, j) {
      const v = this.metrics?.[k]?.values?.[j]?.[i];
      return v === undefined || v === null || Number.isNaN(v) ? null : v;
    },
    isFilled(i, j) {
      return this.metrics.some((m, k) => this.valueAt(k, i, j) !== null);
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
      if (i != this.activeI || j != this.activeJ) {
        return {};
      }
      return {
        outline: this.dark ? 'solid 1px white' : 'solid 1px black',
        outlineOffset: '-1px',
        zIndex: 10,
        ...this.activeStyle,
      };
    },
    rgbFor(k, i, j) {
      const v = this.valueAt(k, i, j);
      const lut = this.luts[k];
      return v === null || !lut ? null : lut.getRGB(v);
    },
    cellStyle(i, j) {
      const style = { width: '100%', height: '100%', overflow: 'hidden' };
      if (this.colorMode !== 'primary') {
        return style;
      }
      const rgb = this.rgbFor(this.primaryIndex, i, j);
      style.background = rgb ? css(rgb) : this.nanColor;
      style.color = rgb ? readableOn(rgb) : this.textOnNan;
      return style;
    },
    bandStyle(k, i, j) {
      const style = {
        height: `${this.bandHeight}px`,
        lineHeight: `${this.bandHeight}px`,
        fontSize: `${this.fontSize}px`,
        textAlign: 'center',
        overflow: 'hidden',
        whiteSpace: 'nowrap',
        fontWeight: 600,
      };
      if (this.colorMode === 'primary') {
        style.borderTop = k ? '1px solid rgba(128, 128, 128, 0.35)' : 'none';
        return style;
      }
      const rgb = this.rgbFor(k, i, j);
      style.background = rgb ? css(rgb) : this.nanColor;
      style.color = rgb ? readableOn(rgb) : this.textOnNan;
      return style;
    },
    label(k, i, j) {
      const v = this.valueAt(k, i, j);
      if (v === null) {
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
    tooltip(i, j) {
      return this.metrics
        .map((m, k) => `${m.label}: ${this.label(k, i, j) || '-'} ${m.units || ''}`.trim())
        .join('\n');
    },
  },
};
