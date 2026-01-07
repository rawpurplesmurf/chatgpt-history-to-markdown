import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { resolve } from "node:path";

test("search UI smoke test", () => {
  const html = readFileSync(
    resolve(process.cwd(), "web", "index.html"),
    "utf8"
  );

  assert.match(html, /class="layout"/);
  assert.match(html, /class="search-panel"/);
  assert.match(html, /class="reader-panel"/);
  assert.match(html, /id="search"/);
  assert.match(html, /id="search-mode"/);
  assert.match(html, /id="search-scope"/);
  assert.match(html, /id="search-role"/);
  assert.match(html, /id="search-sort"/);
  assert.match(html, /id="search-limit"/);
  assert.match(html, /id="search-snippet"/);
  assert.match(html, /id="search-from"/);
  assert.match(html, /id="search-to"/);
  assert.match(html, /id="results"/);
  assert.match(html, /aria-live="polite"/);
  assert.match(html, /role="list"/);
  assert.match(html, /document\.createElement\("button"\)/);
  assert.match(html, /updateHeaderOffset/);
});
