const CHAR_EM = 0.62;

export function formatValue(v, decimals = 2) {
  if (v === null || v === undefined || Number.isNaN(v)) {
    return '';
  }
  if (v === 0) {
    return '0';
  }
  const fixed = Number(v).toFixed(decimals);
  if (Math.abs(v) >= 1e5 || Number(fixed) === 0) {
    return Number(v)
      .toExponential(decimals)
      .replace(/\.?0+e/, 'e');
  }
  return fixed;
}

export function fitFontSize(text, maxSize, availWidth) {
  const em = (text ? text.length : 0) * CHAR_EM;
  return em ? Math.min(maxSize, availWidth / em) : maxSize;
}
