import vtkColorMaps from '@kitware/vtk.js/Rendering/Core/ColorTransferFunction/ColorMaps';
import vtkColorTransferFunction from '@kitware/vtk.js/Rendering/Core/ColorTransferFunction';

if (!vtkColorMaps.getPresetByName('Turbo')) {
  vtkColorMaps.addPreset({
    Name: 'Turbo',
    ColorSpace: 'RGB',
    RGBPoints: [
      0.0,
      0.188,
      0.071,
      0.231,
      0.125,
      0.275,
      0.42,
      0.89,
      0.25,
      0.224,
      0.635,
      0.988,
      0.375,
      0.11,
      0.898,
      0.78,
      0.5,
      0.478,
      0.984,
      0.427,
      0.625,
      0.835,
      0.922,
      0.227,
      0.75,
      0.996,
      0.678,
      0.161,
      0.875,
      0.91,
      0.329,
      0.051,
      1.0,
      0.478,
      0.016,
      0.012,
    ],
  });
}
export class LookupTable {
  constructor(presetName = 'jet', colorRange = [0, 1], nanColor = [1, 1, 1, 1]) {
    this.lookupTable = vtkColorTransferFunction.newInstance();
    this.nanColor = nanColor;
    this.update(presetName, colorRange);
    this.rgba = [0, 0, 0, 0];
  }

  static getPresetNames() {
    return vtkColorMaps.rgbPresetNames;
  }

  update(presetName, colorRange) {
    const preset = vtkColorMaps.getPresetByName(presetName) || vtkColorMaps.getPresetByName('jet');
    this.lookupTable.applyColorMap(preset);
    this.lookupTable.setNanColor(...this.nanColor);
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
    this.nanColor = [r, g, b, a];
    this.lookupTable.setNanColor(r, g, b, a);
  }
}
