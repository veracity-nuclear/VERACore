import { LookupTable } from '../../utils/Colors';
import { toImageURL } from '../../utils/ImageGenerator';

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
    xSizes() {
      this.resize();
    },
    ySizes() {
      this.resize();
    },
    dark() {
      this.updateNanColor();
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
    colorMap() {
      return this.lookupTable.update(this.colorPreset, this.colorRange);
    },
    // One image URL per row instead of one per cell. Each row's cells are
    // concatenated into a single pixel strip and converted once, cutting
    // ~735 canvas/base64/decode operations down to ~49 per update.
    rowImages() {
      const array = this.value;
      const lut = this.colorMap;
      this.dark;

      const urls = [];
      for (let j = 0; j < array.length; j++) {
        const line = array[j];
        // Flatten this row's cells into a single pixel array.
        const rowPixels = line.flat();
        const rowWidth = rowPixels.length;
        urls.push(toImageURL(lut, rowPixels, rowWidth, 1, this.xScale, 1));
      }
      return urls;
    },
    // Cumulative pixel offset of each column / row edge, in layout space
    // (pre-CSS-transform). Index k = left/top edge of cell k.
    // 30px label gutter, gapless cells.
    xOffsets() {
      const offsets = [];
      let acc = 30;
      for (let i = 0; i < this.xSizes.length; i++) {
        offsets.push(acc);
        acc += this.xSizes[i] * this.xScale;
      }
      return offsets;
    },
    yOffsets() {
      const offsets = [];
      let acc = 30;
      for (let j = 0; j < this.ySizes.length; j++) {
        offsets.push(acc);
        acc += this.ySizes[j] * this.yScale;
      }
      return offsets;
    },
    // Total width of the data area (excludes the 30px label gutter).
    rowWidthPx() {
      let w = 0;
      for (let i = 0; i < this.xSizes.length; i++) {
        w += this.xSizes[i] * this.xScale;
      }
      return w;
    },
    highlightStyle() {
      const i = this.activeI;
      const j = this.activeJ;
      if (
        i < 0 || j < 0 ||
        i >= this.xSizes.length || j >= this.ySizes.length
      ) {
        return { display: 'none' };
      }
      return {
        position: 'absolute',
        left: `${this.xOffsets[i]}px`,
        top: `${this.yOffsets[j]}px`,
        width: `${this.xSizes[i] * this.xScale}px`,
        height: `${this.ySizes[j] * this.yScale}px`,
        pointerEvents: 'none',
        ...this.activeStyle,
      };
    },
  },
  created() {
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.lookupTable = new LookupTable(this.colorPreset, this.colorRange);
    this.updateNanColor();
  },
  mounted() {
    this.resizeObserver.observe(this.$el);
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
    resize() {
      const { width, height } = this.$el.getBoundingClientRect();
      let neededWidth = 30 + this.rowWidthPx;
      let neededHeight = 30;
      for (let i = 0; i < this.ySizes.length; i++) {
        neededHeight += this.ySizes[i] * this.yScale;
      }
      const scale = Math.min(width / neededWidth, height / neededHeight);
      this.scaleStyle = { scale };
      this.sizeStyle = {
        width: `${neededWidth + 10}px`,
        height: `${neededHeight + 10}px`,
      };
    },
    // Map an x position within a row's data strip to a column index.
    // The strip starts after the 30px gutter, so shift into layout space
    // before comparing against xOffsets (which include the gutter).
    columnFromX(xInStrip) {
      const xLayout = xInStrip + 30;
      const offsets = this.xOffsets;
      for (let i = offsets.length - 1; i >= 0; i--) {
        if (xLayout >= offsets[i]) {
          return i;
        }
      }
      return 0;
    },
    onRowClick(event, j) {
      // offsetX is relative to the strip element's own box, before the
      // parent CSS scale transform, so it is already in layout space.
      const i = this.columnFromX(event.offsetX);
      this.$emit('click', { i, j });
    },
    onRowHover(event, j) {
      this.activeI = this.columnFromX(event.offsetX);
      this.activeJ = j;
    },
    exit() {
      this.activeI = this.selectedI;
      this.activeJ = this.selectedJ;
    },
    rowUrl(j) {
      return this.rowImages?.[j];
    },
  },
};
