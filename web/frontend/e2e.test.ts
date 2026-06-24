import { chromium, expect, test } from '@playwright/test';

test.describe('llm-infer-cal web UI', () => {
  test('full evaluation flow with builtin model', async () => {
    const browser = await chromium.launch({ headless: true });
    const context = await browser.newContext();
    const page = await context.newPage();

    // Load the page
    await page.goto('http://127.0.0.1:5173/');
    await expect(page.locator('h1')).toContainText('GPU Memory Estimator');

    // Model is initially set to builtin source via default
    // Select Qwen vendor family by clicking the Family dropdown
    await page.click('text=Family');
    await page.click('text=Qwen');

    // Select a specific Qwen model by clicking the Model dropdown
    await page.click('text=Model (');
    await page.click('text=Qwen/Qwen3-30B-A3B');

    // Select H100 GPU by clicking the GPU multi-select
    await page.click('[data-testid="gpu-model-picker"]');
    await page.click('text=NVIDIA');
    await page.click('text=H100');
    await page.keyboard.press('Escape'); // close dropdown

    // Click evaluate
    await page.click('button:has-text("Evaluate")');
