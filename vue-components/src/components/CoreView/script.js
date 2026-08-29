import { LookupTable } from '../../utils/Colors';
import { toImageURL } from '../../utils/ImageGenerator';
import { fitFontSize, formatValue } from '../../utils/format';

const CELL = 30; // unscaled px per assembly
const LABEL_MAX_SIZE = 13; // unscaled px
// A node smaller than this on screen cannot hold a legible number.
const MIN_NODE_PX = 22;
const MAX_LABELS = 4000;
// Beyond nodal (2x2) there are too many values in a cell to label.
const MAX_LABEL_SIDE = 2;

const EMPTY_CORE = { url: null, filled: new Uint8Array(0), rows: 0, cols: 0 };

// Layout is inline rather than class-based: a <style> block on this component
// is not applied by the build, so anything layout-critical must be bound.
const HEADER_STYLE = {
  position: 'absolute',
  width: `${CELL}px`,
  height: `${CELL}px`,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  fontSize: '12px',
  letterSpacing: '0.5px',
  whiteSpace: 'nowrap',
  pointerEvents: 'none',
};

const IMAGE_STYLE = {
  position: 'absolute',
  top: '0',
  left: '0',
  width: '100%',
  height: '100%',
  pointerEvents: 'none',
  imageRendering: 'pixelated',
};

const NOTE_STYLE = {
  position: 'absolute',
  left: '50%',
  bottom: '2px',
  transform: 'translateX(-50%)',
  whiteSpace: 'nowrap',
  pointerEvents: 'none',
  fontSize: '11px',
  opacity: 0.6,
};

const VALUE_STYLE = {
  position: 'absolute',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  whiteSpace: 'nowrap',
  pointerEvents: 'none',
  color: '#000',
  textShadow: '0 0 2px rgba(255,255,255,0.85), 0 0 2px rgba(255,255,255,0.85)',
};

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
    busy: { type: Boolean, default: false },
    aspectRatio: { type: Number, default: 1 },
    dark: { type: Boolean, default: false },
    showLabels: { type: Boolean, default: false },
    decimals: { type: Number, default: 2 },
  },
  data() {
    return {
      activeI: this.selectedI,
      activeJ: this.selectedJ,
      overFilled: false,
      scale: 1,
      // Bumped when the lookup table is mutated outside Vue's reactivity.
      revision: 0,
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
    coreWidth() {
      this.resize();
    },
    coreHeight() {
      this.resize();
    },
    dark() {
      this.updateNanColor();
      this.revision++;
    },
  },
  computed: {
    imageStyle() {
      return IMAGE_STYLE;
    },
    noteStyle() {
      return NOTE_STYLE;
    },
    coreWidth() {
      return this.coreCols || (this.value || []).reduce((m, r) => Math.max(m, r.length), 0);
    },
    coreHeight() {
      return (this.value || []).length;
    },
    colorMap() {
      return this.lookupTable.update(this.colorPreset, this.colorRange);
    },
    // One image for the whole core. Node n sits where the old per-cell call
    // drew it: row-major from the top-left of its assembly.
    core() {
      this.revision;
      const value = this.value || [];
      const rows = this.coreHeight;
      const cols = this.coreWidth;
      const side = this.assemblySize || 1;
      if (!rows || !cols) {
        return EMPTY_CORE;
      }
      const width = cols * side;
      const height = rows * side;
      const pixels = new Float64Array(width * height).fill(NaN);
      const filled = new Uint8Array(rows * cols);
      const nodes = side * side;
      for (let j = 0; j < rows; j++) {
        const line = value[j] || [];
        const stop = Math.min(cols, line.length);
        for (let i = 0; i < stop; i++) {
          const cell = line[i];
          if (!Array.isArray(cell) || !cell.length) {
            continue;
          }
          filled[j * cols + i] = 1;
          const base = j * side * width + i * side;
          for (let n = 0; n < cell.length && n < nodes; n++) {
            const v = cell[n];
            pixels[base + Math.floor(n / side) * width + (n % side)] = Number.isFinite(v) ? v : NaN;
          }
        }
      }
      return { url: toImageURL(this.colorMap, pixels, width, height), filled, rows, cols };
    },
    // Assembly-valued (1x1) and nodal (2x2) cells hold few enough values to
    // label; pin and channel cells do not.
    canLabel() {
      return this.assemblySize > 0 && this.assemblySize <= MAX_LABEL_SIDE;
    },
    showValues() {
      return this.showLabels && this.canLabel;
    },
    labelsHidden() {
      if (!this.showValues) {
        return false;
      }
      const side = this.assemblySize;
      const onScreen = (CELL / side) * Math.min(this.scale, this.scale * (this.aspectRatio || 1));
      const count = this.coreHeight * this.coreWidth * side * side;
      return onScreen < MIN_NODE_PX || count > MAX_LABELS;
    },
    cellLabels() {
      if (!this.showValues || this.labelsHidden) {
        return [];
      }
      const side = this.assemblySize;
      const node = CELL / side;
      const out = [];
      const value = this.value || [];
      for (let j = 0; j < value.length; j++) {
        const line = value[j] || [];
        for (let i = 0; i < line.length; i++) {
          const cell = line[i];
          if (!Array.isArray(cell) || !cell.length) {
            continue;
          }
          for (let n = 0; n < cell.length; n++) {
            const text = formatValue(cell[n], this.decimals);
            if (!text) {
              continue;
            }
            out.push({
              key: `${j}_${i}_${n}`,
              text,
              style: {
                ...VALUE_STYLE,
                left: `${(i * side + (n % side)) * node}px`,
                top: `${(j * side + Math.floor(n / side)) * node}px`,
                width: `${node}px`,
                height: `${node}px`,
                fontSize: `${fitFontSize(text, LABEL_MAX_SIZE, node * 0.9)}px`,
              },
            });
          }
        }
      }
      return out;
    },
    frameStyle() {
      const ar = this.aspectRatio || 1;
      return {
        position: 'absolute',
        top: '50%',
        left: '50%',
        width: `${(this.coreWidth + 1) * CELL}px`,
        height: `${(this.coreHeight + 1) * CELL}px`,
        transform: `translate(-50%, -50%) scale(${ar * this.scale}, ${this.scale})`,
        transformOrigin: 'center center',
      };
    },
    dataStyle() {
      return {
        position: 'absolute',
        left: `${CELL}px`,
        top: `${CELL}px`,
        width: `${this.coreWidth * CELL}px`,
        height: `${this.coreHeight * CELL}px`,
      };
    },
    // Gridline thickness is inverted through the frame scale so a line lands
    // on about one device pixel instead of smearing when the core shrinks.
    lineWidths() {
      const ar = this.aspectRatio || 1;
      const t = this.scale || 1;
      return { v: Math.max(1, 1 / (ar * t)), h: Math.max(1, 1 / t) };
    },
    gridStyle() {
      const { v, h } = this.lineWidths;
      const c = this.dark ? 'rgba(255,255,255,0.35)' : 'rgba(0,0,0,0.25)';
      return {
        position: 'absolute',
        top: '0',
        left: '0',
        width: '100%',
        height: '100%',
        boxSizing: 'border-box',
        pointerEvents: 'none',
        borderRight: `${v}px solid ${c}`,
        borderBottom: `${h}px solid ${c}`,
        backgroundImage: [
          `repeating-linear-gradient(to right, ${c} 0, ${c} ${v}px,` +
            ` transparent ${v}px, transparent ${CELL}px)`,
          `repeating-linear-gradient(to bottom, ${c} 0, ${c} ${h}px,` +
            ` transparent ${h}px, transparent ${CELL}px)`,
        ].join(','),
      };
    },
    highlightStyle() {
      const { rows, cols } = this.core;
      if (this.activeI < 0 || this.activeJ < 0 || this.activeI >= cols || this.activeJ >= rows) {
        return { display: 'none' };
      }
      const { v, h } = this.lineWidths;
      return {
        position: 'absolute',
        pointerEvents: 'none',
        left: `${this.activeI * CELL}px`,
        top: `${this.activeJ * CELL}px`,
        width: `${CELL}px`,
        height: `${CELL}px`,
        boxSizing: 'border-box',
        borderStyle: 'solid',
        borderWidth: `${h}px ${v}px`,
        borderColor: this.dark ? 'white' : 'black',
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
      if (width < 1 || height < 1 || !this.coreWidth || !this.coreHeight) {
        return;
      }
      const ar = this.aspectRatio || 1;
      this.scale = Math.min(
        width / ((this.coreWidth + 1) * CELL * ar),
        height / ((this.coreHeight + 1) * CELL)
      );
    },
    headerStyle(offset, active) {
      return {
        ...HEADER_STYLE,
        ...offset,
        opacity: active ? 1 : 0.75,
        fontWeight: active ? 700 : 400,
      };
    },
    xLabelStyle(i) {
      return this.headerStyle({ left: `${(i + 1) * CELL}px`, top: '0' }, i === this.activeI);
    },
    yLabelStyle(j) {
      return this.headerStyle({ left: '0', top: `${(j + 1) * CELL}px` }, j === this.activeJ);
    },
    isFilled(i, j) {
      const { filled, rows, cols } = this.core;
      return i >= 0 && j >= 0 && i < cols && j < rows && filled[j * cols + i] === 1;
    },
    // offsetX/offsetY are in the data layer's own untransformed coordinate
    // space, so the frame scale does not need to be undone here.
    cellAt(event) {
      const i = Math.floor(event.offsetX / CELL);
      const j = Math.floor(event.offsetY / CELL);
      return this.isFilled(i, j) ? { i, j } : null;
    },
    onHover(event) {
      const cell = this.cellAt(event);
      this.overFilled = !!cell;
      if (!cell) {
        this.exit();
        return;
      }
      this.activeI = cell.i;
      this.activeJ = cell.j;
    },
    onClick(event) {
      const cell = this.cellAt(event);
      if (cell) {
        this.$emit('click', cell);
      }
    },
    exit() {
      this.activeI = this.selectedI;
      this.activeJ = this.selectedJ;
      this.overFilled = false;
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
