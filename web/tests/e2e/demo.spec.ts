import { test, expect } from "@playwright/test";

test("demo: pick a player and see the roster grow", async ({ page }) => {
  await page.goto("/demo");
  // bots may pick first; wait until it is our turn and recs render
  const draftButtons = page.getByRole("button", { name: "Draft" });
  await expect(draftButtons.first()).toBeVisible({ timeout: 30_000 });
  await draftButtons.first().click();
  await expect(page.getByText(/My roster \(1\)/)).toBeVisible({ timeout: 30_000 });
});
