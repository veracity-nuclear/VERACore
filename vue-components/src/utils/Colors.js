import vtkColorMaps from '@kitware/vtk.js/Rendering/Core/ColorTransferFunction/ColorMaps';
import vtkColorTransferFunction from '@kitware/vtk.js/Rendering/Core/ColorTransferFunction';

export class LookupTable {
  constructor(presetName = 'jet', colorRange = [0, 1], nanColor = [1, 1, 1, 1]) {
    this.lookupTable = vtkColorTransferFunction.newInstance();
    this.lookupTable.setNanColor(...nanColor);
    this.update(presetName, colorRange);
    this.rgba = [0, 0, 0, 0];
  }
  static getPresetNames() {
    return vtkColorMaps.rgbPresetNames;
  }

  update(presetName, colorRange) {
    const preset = vtkColorMaps.getPresetByName(presetName);
    this.lookupTable.applyColorMap(preset);
    const [lo, hi] = this.safeRange(colorRange);
    this.lookupTable.setMappingRange(lo, hi);
    this.lookupTable.updateRange();
    return this;
  }

  safeRange(colorRange) {
    const lo = Number(colorRange[0]);
    const hi = Number(colorRange[1]);
    if (!Number.isFinite(lo) || !Number.isFinite(hi)) {
      return [0, 1];
    }
    if (hi - lo > 0) {
      return [lo, hi];
    }
    const pad = Math.abs(lo) * 1e-6 || 1e-6;
    return [lo - pad, lo + pad];
  }

  applyRGBA(value, offset, array) {
    this.lookupTable.getColor(value, this.rgba);
    array[offset] = this.rgba[0] * 255 + 0.5;
    array[offset + 1] = this.rgba[1] * 255 + 0.5;
    array[offset + 2] = this.rgba[2] * 255 + 0.5;
    array[offset + 3] = 255;

    return offset + 4;
  }
  getRGB(value) {
    this.lookupTable.getColor(value, this.rgba);
    return [this.rgba[0], this.rgba[1], this.rgba[2]];
  }
  setNanColor(r, g, b, a) {
    this.lookupTable.setNanColor(r, g, b, a);
  }
}
