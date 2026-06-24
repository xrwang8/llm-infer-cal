# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: e2e.test.ts >> llm-infer-cal web UI >> full evaluation flow with builtin model
- Location: e2e.test.ts:4:3

# Error details

```
Error: expect(locator).toContainText(expected) failed

Locator: locator('h1')
Expected substring: "llm-infer-cal"
Received string:    "GPU Memory Estimator"
Timeout: 5000ms

Call log:
  - Expect "toContainText" with timeout 5000ms
  - waiting for locator('h1')
    14 × locator resolved to <h1>GPU Memory Estimator</h1>
       - unexpected value "GPU Memory Estimator"

```

```yaml
- heading "GPU Memory Estimator" [level=1]
```

# Test source

```ts
  1  | import { chromium, expect, test } from '@playwright/test';
  2  | 
  3  | test.describe('llm-infer-cal web UI', () => {
  4  |   test('full evaluation flow with builtin model', async () => {
  5  |     const browser = await chromium.launch();
  6  |     const context = await browser.newContext();
  7  |     const page = await context.newPage();
  8  | 
  9  |     // Load the page
  10 |     await page.goto('http://127.0.0.1:5173/');
> 11 |     await expect(page.locator('h1')).toContainText('llm-infer-cal');
     |                                      ^ Error: expect(locator).toContainText(expected) failed
  12 | 
  13 |     // Select a builtin model (Qwen/Qwen3-30B-A3B is in the catalog)
  14 |     await page.selectOption('select[data-testid="model-source-picker"]', 'builtin');
  15 |     await page.fill('input[data-testid="model-id-input"]', 'Qwen/Qwen3-30B-A3B');
  16 | 
  17 |     // Select a GPU
  18 |     const gpuPicker = page.locator('select[data-testid="gpu-model-picker"]');
  19 |     await gpuPicker.selectOption('H100');
  20 | 
  21 |     // Click evaluate
  22 |     await page.click('button:has-text("Evaluate")');
  23 | 
  24 |     // Wait for results
  25 |     await page.waitForSelector('text=Total VRAM', { timeout: 15000 });
  26 |     await page.waitForSelector('text=Model Weights / GPU', { timeout: 5000 });
  27 | 
  28 |     // Check the report card is populated
  29 |     const totalVramCard = page.locator('text=Total VRAM').locator('..').locator('..');
  30 |     await expect(totalVramCard).toContainText('GB');
  31 | 
  32 |     // Check model weights are shown
  33 |     await expect(page.locator('text=Model Weights / GPU')).toBeVisible();
  34 | 
  35 |     // Check recommended GPU count is displayed
  36 |     await expect(page.locator('text=Recommended GPUs')).toBeVisible();
  37 | 
  38 |     // Check explain section renders
  39 |     await expect(page.locator('text=Weight bytes')).toBeVisible();
  40 | 
  41 |     // Check VRAM breakdown section
  42 |     await expect(page.locator('text=VRAM Breakdown')).toBeVisible();
  43 | 
  44 |     // Clean up
  45 |     await browser.close();
  46 |   });
  47 | 
  48 |   test('multi-GPU comparison', async () => {
  49 |     const browser = await chromium.launch();
  50 |     const context = await browser.newContext();
  51 |     const page = await context.newPage();
  52 | 
  53 |     await page.goto('http://127.0.0.1:5173/');
  54 | 
  55 |     // Select builtin model
  56 |     await page.selectOption('select[data-testid="model-source-picker"]', 'builtin');
  57 |     await page.fill('input[data-testid="model-id-input"]', 'Qwen/Qwen2.5-72B-Instruct');
  58 | 
  59 |     // Select multiple GPUs (H100, A100-80G, H800)
  60 |     const gpuPicker = page.locator('select[data-testid="gpu-model-picker"]');
  61 |     await gpuPicker.selectOption(['H100', 'A100-80G', 'H800']);
  62 | 
  63 |     // Evaluate
  64 |     await page.click('button:has-text("Evaluate")');
  65 | 
  66 |     // Wait for comparison table
  67 |     await page.waitForSelector('table', { timeout: 15000 });
  68 | 
  69 |     // Check that at least 3 rows exist (header + 3 GPUs)
  70 |     const rows = await page.locator('table tr').count();
  71 |     expect(rows).toBeGreaterThanOrEqual(4);
  72 | 
  73 |     await browser.close();
  74 |   });
  75 | });
  76 | 
```