import vtkColorMaps from '@kitware/vtk.js/Rendering/Core/ColorTransferFunction/ColorMaps';
import vtkColorTransferFunction from '@kitware/vtk.js/Rendering/Core/ColorTransferFunction';

// Face order per cell: WEST, NORTH, EAST, SOUTH.
const WEST = 0;
const NORTH = 1;
const EAST = 2;
const SOUTH = 3;

export default {
  name: 'VeraAssemblySurface',
  props: {
    // value[idx] = [w, n, e, s] for each pin/node cell, row-major.
    value: {
      type: Array,
      default: () => [],
    },
    selectedI: { type: Number, default: -1 },
    selectedJ: { type: Number, default: -1 },
    selectedSurface: { type: Number, default: -1 },
    colorPreset: { type: String, default: 'jet' },
    colorRange: { type: Array, default: () => [0, 1] },
    activeStyle: {
      type: Object,
      default: () => ({}),
    },
    busy: { type: Boolean, default: false },
    dark: { type: Boolean, default: false },
    cellSize: { type: Number, default: 60 },
  },
  watch: {
    selectedI(i) { this.activeI = i + 1; },
    selectedJ(j) { this.activeJ = j + 1; },
    selectedSurface(s) { this.activeSurface = s; },
    value() { this.updateColors(); },
    colorPreset() { this.updateLookupTable(); this.updateColors(); },
    colorRange() { this.updateLookupTable(); this.updateColors(); },
    dark() { this.updateLookupTable(); this.updateColors(); },
    sideCount() { this.resize(); },
  },
  data() {
    return {
      activeI: this.selectedI + 1,
      activeJ: this.selectedJ + 1,
      activeSurface: this.selectedSurface,
      // colors[idx] = [wColor, nColor, eColor, sColor]
      colors: [],
      sizeStyle: { width: '100px', height: '100px' },
      scaleStyle: { scale: 0.5 },
    };
  },
  computed: {
    sideCount() {
      return Math.round(Math.sqrt(this.value.length));
    },
    // Triangle vertex point-strings in cellSize space (NORTH on top).
    trianglePoints() {
      const s = this.cellSize;
      const c = `${s / 2},${s / 2}`;
      return {
        [NORTH]: `0,0 ${s},0 ${c}`,
        [SOUTH]: `0,${s} ${s},${s} ${c}`,
        [WEST]: `0,0 0,${s} ${c}`,
        [EAST]: `${s},0 ${s},${s} ${c}`,
      };
    },
    // Text anchor point (roughly the centroid) of each triangle.
    triangleCentroids() {
      const s = this.cellSize;
      return {
        [NORTH]: [s / 2, s / 6],
        [SOUTH]: [s / 2, (s * 5) / 6],
        [WEST]: [s / 6, s / 2],
        [EAST]: [(s * 5) / 6, s / 2],
      };
    },
  },
  created() {
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.lookupTable = vtkColorTransferFunction.newInstance();
    this.updateLookupTable();
    this.updateColors();
  },
  mounted() {
    this.resizeObserver.observe(this.$el);
  },
  beforeDestroy() {
    this.resizeObserver.disconnect();
    this.resizeObserver = null;
  },
  methods: {
    resize() {
      const { width, height } = this.$el.getBoundingClientRect();
      const neededSize = (this.sideCount + 1) * 35;
      const availableSpace = Math.min(width, height);
      const scale = availableSpace / neededSize;
      this.scaleStyle = { scale };
      this.sizeStyle = {
        width: `${neededSize + 10}px`,
        height: `${neededSize + 10}px`,
      };
    },
    toIdx(i, j) {
      return (j - 1) * this.sideCount + (i - 1);
    },
    hasCell(i, j) {
      const cell = this.value[this.toIdx(i, j)];
      return Array.isArray(cell) && cell.length >= 4;
    },
    // Which triangle (surface) a cursor position within a cell falls in.
    // xf, yf are 0..1 fractions of the cell. The two diagonals split it.
    surfaceAt(xf, yf) {
      const belowMainDiag = yf > xf; // '\' diagonal (0,0)->(1,1)
      const belowAntiDiag = yf > 1 - xf; // '/' diagonal (1,0)->(0,1)
      if (!belowMainDiag && !belowAntiDiag) return NORTH;
      if (belowMainDiag && belowAntiDiag) return SOUTH;
      if (belowMainDiag && !belowAntiDiag) return WEST;
      return EAST;
    },
    onCellHover(event, i, j) {
      this.activeI = i;
      this.activeJ = j;
    },
    onCellClick(event, i, j) {
      const rect = event.currentTarget.getBoundingClientRect();
      const xf = (event.clientX - rect.left) / rect.width;
      const yf = (event.clientY - rect.top) / rect.height;
      const surface = this.surfaceAt(xf, yf);
      this.$emit('click', { i: i - 1, j: j - 1, surface });
    },
    exit() {
      this.activeI = this.selectedI + 1;
      this.activeJ = this.selectedJ + 1;
    },
    toStyle(i, j) {
      if (i !== this.activeI || j !== this.activeJ) {
        return {};
      }
      return {
        outline: this.dark ? 'solid 1px white' : 'solid 1px black',
        outlineOffset: '-1px',
        zIndex: 10,
        ...this.activeStyle,
      };
    },
    isSelectedTriangle(i, j, surface) {
      return (
        i === this.activeI &&
        j === this.activeJ &&
        surface === this.activeSurface
      );
    },
    triangleFill(i, j, surface) {
      const cellColors = this.colors[this.toIdx(i, j)];
      if (!cellColors) return this.dark ? 'rgb(30,30,30)' : 'rgb(255,255,255)';
      return cellColors[surface];
    },
    faceValue(i, j, surface) {
      const cell = this.value[this.toIdx(i, j)];
      if (!Array.isArray(cell) || cell.length < 4) return null;
      return cell[surface];
    },
    faceText(i, j, surface) {
      const v = this.faceValue(i, j, surface);
      if (v === null || v === undefined || Number.isNaN(v)) return '';
      if (v === 0) return '0';
      const abs = Math.abs(v);
      if (abs < 1e-2 || abs >= 1e5) {
        return v.toExponential(1).replace(/\.?0+e/, 'e');
      }
      return v.toFixed(2);
    },
    updateLookupTable() {
      const preset = vtkColorMaps.getPresetByName(this.colorPreset);
      this.lookupTable.applyColorMap(preset);
      this.lookupTable.setMappingRange(this.colorRange[0], this.colorRange[1]);
      this.lookupTable.updateRange();
      if (this.dark) {
        this.lookupTable.setNanColor(30 / 255, 30 / 255, 30 / 255, 1);
      } else {
        this.lookupTable.setNanColor(1, 1, 1, 1);
      }
    },
    colorStr(v) {
      const rgb = [];
      this.lookupTable.getColor(v, rgb);
      const r = Math.floor(255 * rgb[0] + 0.5);
      const g = Math.floor(255 * rgb[1] + 0.5);
      const b = Math.floor(255 * rgb[2] + 0.5);
      return `rgb(${r},${g},${b})`;
    },
    updateColors() {
      const out = [];
      for (let k = 0; k < this.value.length; k++) {
        const cell = this.value[k];
        if (!Array.isArray(cell) || cell.length < 4) {
          out.push(null);
          continue;
        }
        out.push([
          this.colorStr(cell[WEST]),
          this.colorStr(cell[NORTH]),
          this.colorStr(cell[EAST]),
          this.colorStr(cell[SOUTH]),
        ]);
      }
      this.colors = out;
    },
  },
};