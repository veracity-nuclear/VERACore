import { LookupTable } from '../../utils/Colors';
import { toImageURL } from '../../utils/ImageGenerator';

function simplifyNumber(v, targetSize = 6) {
  if (!Number.isFinite(v)) return v;
  const abs = Math.abs(v);
  if (v !== 0 && (abs < 1e-3 || abs >= 1e5)) {
    return v;
  }
  let strValue = `${v}`;
  let precision = targetSize;
  while (strValue.length > 6 && precision > 0) {
    precision -= 1;
    strValue = n.toFixed(precision);
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
      const min = Number(this.value?.[0]);
      const max = Number(this.value?.[1]);
      const fallback = Number.isFinite(max) ? max : (Number.isFinite(min) ? min : 0);

      if (!Number.isFinite(min) || !Number.isFinite(max) || max <= min) {
        return toImageURL(this.colorMap, [fallback], 1, 1);
      }

      const steps = 512;
      const delta = (max - min) / steps;
      const samples = new Array(steps + 1);
      for (let k = 0; k <= steps; k++) {
        samples[k] = max - k * delta;
      }

      // width=1, height=samples.length: a tall 1-pixel-wide strip
      return toImageURL(this.colorMap, samples, 1, samples.length);
    }
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