import vtkColorMaps from '@kitware/vtk.js/Rendering/Core/ColorTransferFunction/ColorMaps';
import vtkColorTransferFunction from '@kitware/vtk.js/Rendering/Core/ColorTransferFunction';

export default {
  name: 'VeraAssembly',
  props: {
    value: {
      type: Array,
      default: () => [0, 1, 1, 0],
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
      default: () => ({}),
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
      this.activeI = i + 1;
    },
    selectedJ(j) {
      this.activeJ = j + 1;
    },
    value() {
      this.updateColors();
    },
    colorPreset() {
      this.updateLookupTable();
      this.updateColors();
    },
    colorRange() {
      this.updateLookupTable();
      this.updateColors();
    },
    dark() {
      this.updateLookupTable();
      this.updateColors();
    },
    sideCount() {
      this.resize();
    },
  },
  data() {
    return {
      activeI: this.selectedI + 1,
      activeJ: this.selectedJ + 1,
      colors: [],
      sizeStyle: { width: '100px', height: '100px' },
      scaleStyle: { scale: 0.5 },
    };
  },
  computed: {
    sideCount() {
      return Math.sqrt(this.value.length);
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
    hover(i, j) {
      this.activeI = i;
      this.activeJ = j;
    },
    exit() {
      this.activeI = this.selectedI + 1;
      this.activeJ = this.selectedJ + 1;
    },
    toIdx(i, j) {
      return (j - 1) * this.sideCount + (i - 1);
    },
    toValue(i, j) {
      const v = this.value[this.toIdx(i, j)];
      if (v === undefined || Number.isNaN(v)) {
        return '';
      }
      if (v === 0) {
        return '0';
      }
      const abs = Math.abs(v);
      if (abs < 1e-2 || abs >= 1e5) {
        return v.toExponential(2).replace(/\.?0+e/, 'e');
      }
      return v.toFixed(2);
    },
    toStyle(i, j) {
      if (i != this.activeI || j != this.activeJ) {
        return {};
      }
      return {
        outline: this.dark ? 'solid 3px white' : 'solid 3px black',
        outlineOffset: '-1px',
        zIndex: 10,
        ...this.activeStyle,
      };
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
    updateColors() {
      const colors = [];
      const rgb = [];
      for (let i = 0; i < this.value.length; i++) {
        const v = this.value[i];
        this.lookupTable.getColor(v, rgb);
        colors.push(
          `rgb(${Math.floor(255.0 * rgb[0] + 0.5)}, ${Math.floor(
            255.0 * rgb[1] + 0.5
          )}, ${Math.floor(255.0 * rgb[2] + 0.5)})`
        );
      }

      this.colors = colors;
    },
  },
};