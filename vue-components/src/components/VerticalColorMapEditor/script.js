import { LookupTable } from '../../utils/Colors';
import { toImageURL } from '../../utils/ImageGenerator';

function simplifyNumber(v, targetSize = 6) {
  let strValue = `${v}`;
  let precision = targetSize;
  while (strValue.length > 6) {
    precision -= 1;
    strValue = v.toFixed(precision);
  }
  return Number(strValue);
}

export default {
  name: 'VeraVerticalColorMapEditor',
  props: {
    value: {
      type: Array,
      default: () => [0, 1],
    },
    colorPreset: {
      type: String,
      default: 'erdc_rainbow_bright',
    },
  },
  data() {
    return {
      minValue: simplifyNumber(this.value[0]),
      maxValue: simplifyNumber(this.value[1]),
    };
  },
  watch: {
    value() {
      this.minValue = simplifyNumber(this.value[0]);
      this.maxValue = simplifyNumber(this.value[1]);
    },
  },
  computed: {
    colorMap() {
      return this.lookupTable.update(this.colorPreset, this.value);
    },
    imgSrc() {
      // samples ordered high -> low so the rendered image reads top=max, bottom=min
      const samples = [];
      const delta = (this.value[1] - this.value[0]) / 512;
      let v = this.value[1];
      while (v > this.value[0]) {
        samples.push(v);
        v -= delta;
      }
      // width=1, height=samples.length: a tall 1-pixel-wide strip
      return toImageURL(this.colorMap, samples, 1, samples.length);
    },
  },
  created() {
    this.lookupTable = new LookupTable(this.colorPreset, this.value);
  },
  methods: {
    validateRange() {
      let error = 0;
      const v0 = Number(this.minValue);
      if (Number.isNaN(v0)) {
        this.minValue = simplifyNumber(this.value[0]);
        error++;
      }
      const v1 = Number(this.maxValue);
      if (Number.isNaN(v1)) {
        this.maxValue = simplifyNumber(this.value[1]);
        error++;
      }
      if (error === 0) {
        this.$emit('input', [v0, v1]);
      }
    },
  },
};