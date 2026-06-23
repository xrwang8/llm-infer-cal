import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import { App } from './App';

describe('App shell', () => {
  it('does not render header status metrics', () => {
    const html = renderToStaticMarkup(<App />);

    expect(html).not.toContain('statusRail');
    expect(html).not.toContain('127.0.0.1:8080');
  });

  it('renders the GPU model picker as a multi-select control', () => {
    const html = renderToStaticMarkup(<App />);

    expect(html).toContain('data-testid="gpu-model-picker"');
  });

  it('does not render explain or refresh cache controls', () => {
    const html = renderToStaticMarkup(<App />);

    expect(html).not.toContain('输出推导链（--explain）');
    expect(html).not.toContain('刷新缓存');
  });

  it('renders reference-inspired calculator sections', () => {
    const html = renderToStaticMarkup(<App />);

    expect(html).toContain('VRAM Breakdown');
    expect(html).toContain('Formula Reference');
    expect(html).toContain('Inference Optimizations');
    expect(html).toContain('GPU 对比');
  });
});
