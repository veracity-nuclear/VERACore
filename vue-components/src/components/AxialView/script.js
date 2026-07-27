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
    letterSpacing: {
      type: Number,
      default: 0.5,
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
    };
  },
  computed: {
    rowCount() {
      return Math.min(this.value.length, this.ySizes.length);
    },
    columnCount() {
      return this.xSizes.length;
    },
    nanColor() {
      const level = this.dark ? 30 / 255 : 1;
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
      for (let j = 0; j < this.rowCount; j += 1) {
        if (this.rowImages[j]) {
          rows.push({ j, url: this.rowImages[j] });
        }
      }
      return rows;
    },
    gutterWidth() {
      let width = 0;
      for (let j = 0; j < this.rowCount; j += 1) {
        width = Math.max(width, this.textWidth(this.yLabels[j]));
      }
      return width ? Math.ceil(width) + this.labelPadding : 0;
    },
    headerHeight() {
      let width = 0;
      for (let i = 0; i < this.columnCount; i += 1) {
        width = Math.max(width, this.textWidth(this.xLabels[i]));
      }
      return width ? this.lineHeight : 0;
    },
    lineHeight() {
      return Math.ceil(this.fontSize * 1.4);
    },
    xLabelGap() {
      let width = 0;
      for (let i = 0; i < this.columnCount; i += 1) {
        width = Math.max(width, this.textWidth(this.xLabels[i]));
      }
      return Math.ceil(width) + this.labelPadding;
    },
    yLabelGap() {
      return this.lineHeight;
    },
    availWidth() {
      return Math.max(0, this.boxWidth - this.gutterWidth);
    },
    availHeight() {
      return Math.max(0, this.boxHeight - this.headerHeight);
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
      return Math.min(
        this.availWidth / this.unitWidth,
        this.availHeight / this.unitHeight,
        this.maxCellSize / (Math.max(...this.xSizes) * this.xScale),
        this.maxCellSize / (Math.max(...this.ySizes) * this.yScale),
      );
    },
    xEdges() { return edgesFrom(this.xSizes, this.xScale * this.fit); },
    yEdges() { return edgesFrom(this.ySizes, this.yScale * this.fit); },
    dataWidth() {
      return this.xEdges[this.xEdges.length - 1];
    },
    dataHeight() {
      return this.yEdges[this.yEdges.length - 1];
    },
    visibleColumns() {
      return decimate(centersOf(this.xEdges), this.xLabelGap);
    },
    visibleRows() {
      return decimate(centersOf(this.yEdges).slice(0, this.rowCount), this.yLabelGap);
    },
    frameStyle() {
      const width = this.gutterWidth + this.dataWidth;
      const height = this.headerHeight + this.dataHeight;
      return {
        position: 'absolute',
        left: `${Math.round((this.boxWidth - width) / 2)}px`,
        top: `${Math.round((this.boxHeight - height) / 2)}px`,
        width: `${width}px`,
        height: `${height}px`,
        fontSize: `${this.fontSize}px`,
        letterSpacing: `${this.letterSpacing}px`,
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
  },
  created() {
    this.lookupTable = new LookupTable(this.colorPreset, this.colorRange);
    this.measureContext = document.createElement('canvas').getContext('2d');
    this.resizeObserver = new ResizeObserver(() => this.resize());
  },
  mounted() {
    this.fontFamily = window.getComputedStyle(this.$el).fontFamily || 'sans-serif';
    this.resizeObserver.observe(this.$el);
    this.$nextTick(() => this.resize());
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
    textWidth(text) {
      const value = text == null ? '' : String(text);
      if (!value) {
        return 0;
      }
      this.measureContext.font = `${this.fontSize}px ${this.fontFamily}`;
      return this.measureContext.measureText(value).width + this.letterSpacing * value.length;
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
    xLabelStyle(i) {
      return {
        position: 'absolute',
        left: `${this.gutterWidth + this.xEdges[i]}px`,
        top: 0,
        width: `${this.xEdges[i + 1] - this.xEdges[i]}px`,
        height: `${this.headerHeight}px`,
      };
    },
    yLabelStyle(j) {
      return {
        position: 'absolute',
        left: 0,
        top: `${this.headerHeight + this.yEdges[j]}px`,
        width: `${Math.max(0, this.gutterWidth - this.labelPadding)}px`,
        height: `${this.yEdges[j + 1] - this.yEdges[j]}px`,
      };
    },
    locate(event) {
      const rect = event.currentTarget.getBoundingClientRect();
      return {
        i: indexAt(this.xEdges, event.clientX - rect.left),
        j: indexAt(this.yEdges, event.clientY - rect.top),
      };
    },
    onClick(event) {
      this.$emit('click', this.locate(event));
    },
    onHover(event) {
      const { i, j } = this.locate(event);
      this.activeI = i;
      this.activeJ = j;
    },
    exit() {
      this.activeI = this.selectedI;
      this.activeJ = this.selectedJ;
    },
  },
};