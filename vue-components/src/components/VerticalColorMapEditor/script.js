import { LookupTable } from '../../utils/Colors';
import { toImageURL } from '../../utils/ImageGenerator';

function simplifyNumber(v, targetSize = 6) {
  if (!Number.isFinite(v)) return v;
  if (v === 0) return 0;
  const abs = Math.abs(v);
  if (abs < 1e-3 || abs >= 1e5) {
    return v.toExponential(2).replace(/\.?0+e/, 'e').replace('e+', 'e');
  }
  // Normal range: trim decimals to fit the field.
  let strValue = `${v}`;
  let precision = targetSize;
  while (strValue.length > 6 && precision > 0) {
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
    units: { 
      type: String, 
      default: '' ,
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
      const lo = this.value[0];
      const hi = this.value[1];
      const N = 512;
      const samples = [];
      if (hi > lo) {
        for (let k = 0; k < N; k++) {
          samples.push(hi - (k / (N - 1)) * (hi - lo));
        }
      } else {
        samples.push(lo);
      }
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