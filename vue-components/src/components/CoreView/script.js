import { LookupTable } from '../../utils/Colors';
import { toImageURL } from '../../utils/ImageGenerator';

const CELL = 30;

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
    labels: { type: Array, default: () => [] },
    aspectRatio: { type: Number, default: 1 },
    dark: { type: Boolean, default: false },
  },
  watch: {
    selectedI(i) { this.activeI = i; },
    selectedJ(j) { this.activeJ = j; },
    aspectRatio() { this.resize(); },
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
    }
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
    toLabel(i, j) {
      const v = this.labels?.[j]?.[i];
      if (v === undefined || v === null || Number.isNaN(v)) {
        return '';
      }
      return Number(v).toFixed(2);
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