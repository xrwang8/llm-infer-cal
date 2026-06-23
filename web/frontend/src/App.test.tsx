import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import { App } from './App';

describe('App shell', () => {
  it('does not render header status metrics', () => {
    const html = renderToStaticMarkup(<App />);

    expect(html).not.toContain('statusRail');
    expect(html).not.toContain('127.0.0.1:8080');
  });
});
