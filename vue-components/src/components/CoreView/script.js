import { LookupTable } from '../../utils/Colors';
import { toImageURL } from '../../utils/ImageGenerator';

const CELL = 30;
// Labels are laid out in their own coordinate system, LABEL_UNIT per node.
// Font sizes near 1.0 give browsers wrong glyph metrics, so keep them large
// and let the viewBox scale them down.
const LABEL_UNIT = 100;
const LABEL_MAX_SIZE = 26; // cap, so short labels do not fill the cell
const LABEL_WIDTH = 86; // width a label may occupy, leaving a margin
const CHAR_EM = 0.62; // approximate digit advance, in em
// Beyond nodal (2x2) there are too many values in a cell to label.
const MAX_LABEL_SIDE = 2;

export default {
  name: 'VeraCore',
  props: {
    value: {
      type: Array,
      default: () => [[[], [], []], [[], []], [[]]],
    },
    selectedI: { type: Number, default: -1 },
    selectedJ: { type: Number, default: -1 },
    colorPreset: { type: String, default: 'erdc_rainbow_bright' },
    colorRange: { type: Array, default: () => [0, 1] },
    activeStyle: {
      type: Object,
      default: () => ({}),
    },
    xLabels: { type: Array, default: () => ['H', 'G', 'F', 'E', 'D', 'C', 'B', 'A'] },
    yLabels: { type: Array, default: () => ['8', '9', '10', '11', '12', '13', '14', '15'] },
    assemblySize: { type: Number, default: 0 },
    coreCols: { type: Number, default: 0 },
    scaling: { type: Number, default: 2 },
    busy: { type: Boolean, default: false },
    aspectRatio: { type: Number, default: 1 },
    dark: { type: Boolean, default: false },
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
    dark() {
      this.updateNanColor();
      this.imagesReady++;
    },
    value() {
      this.$nextTick(() => {
        this.imagesReady++;
        this.resize();
      });
    },
  },
  data() {
    return {
      activeI: this.selectedI,
      activeJ: this.selectedJ,
      sizeStyle: { width: '100px', height: '100px' },
      scaleStyle: { scale: 1 },
      imagesReady: 0,
    };
  },
  computed: {
    coreWidth() {
      return this.coreCols || (this.value || []).reduce((m, r) => Math.max(m, r.length), 0);
    },
    assemblyWidth() {
      return this.assemblySize;
    },
    // Assembly-valued (1x1) and nodal (2x2) cells hold few enough values to
    // label; pin and channel cells do not.
    canLabel() {
      return this.assemblySize > 0 && this.assemblySize <= MAX_LABEL_SIDE;
    },
    showValues() {
      return this.showLabels && this.canLabel;
    },
    labelViewBox() {
      const span = this.assemblySize * LABEL_UNIT;
      return `0 0 ${span} ${span}`;
    },
    colorMap() {
      return this.lookupTable.update(this.colorPreset, this.colorRange);
    },
    images() {
      this.imagesReady;
      const array = this.value || [];
      const lut = this.colorMap;
      const width = this.assemblyWidth;
      const images = [];
      for (let j = 0; j < array.length; j++) {
        const line = array[j] || [];
        const lineImages = [];
        images.push(lineImages);
        for (let i = 0; i < line.length; i++) {
          const cell = line[i];
          lineImages.push(
            Array.isArray(cell) && cell.length
              ? toImageURL(lut, cell, width, width, this.scaling, this.scaling)
              : null
          );
        }
      }
      return images;
    },
    // labels[j][i] = one entry per node, positioned in node units. Node n
    // sits where toImageURL draws it: row-major from the top-left.
    labels() {
      if (!this.showValues) {
        return [];
      }
      const side = this.assemblySize;
      return (this.value || []).map((line) =>
        (line || []).map((cell) => {
          if (!Array.isArray(cell) || !cell.length) {
            return [];
          }
          return cell
            .map((v, n) => {
              const text = this.toLabel(v);
              return {
                text,
                x: ((n % side) + 0.5) * LABEL_UNIT,
                y: (Math.floor(n / side) + 0.5) * LABEL_UNIT,
                size: Math.min(LABEL_MAX_SIZE, LABEL_WIDTH / (text.length * CHAR_EM)),
              };
            })
            .filter((label) => label.text);
        })
      );
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
    isFilled(i, j) {
      const cell = this.value?.[j]?.[i];
      return Array.isArray(cell) && cell.length > 0;
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
    toUrl(i, j) {
      return this.images?.[j]?.[i];
    },
    toLabel(v) {
      if (v === undefined || v === null || Number.isNaN(v)) {
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
    updateNanColor() {
      if (this.dark) {
        this.lookupTable.setNanColor(30 / 255, 30 / 255, 30 / 255, 1);
      } else {
        this.lookupTable.setNanColor(1, 1, 1, 1);
      }
    },
  },
};
