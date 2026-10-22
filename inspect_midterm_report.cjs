const fs = require('fs');
const path = require('path');
const {pathToFileURL, fileURLToPath} = require('url');
const {chromium} = require('C:/Users/zzczz/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');

(async () => {
  const folder = path.resolve(process.argv[2] || 'results/midterm-access-v2');
  const browser = await chromium.launch({channel:'msedge',headless:true});
  const checks = [];
  try {
    for (const [name,width,height] of [['desktop',1440,1000],['mobile',390,844]]) {
      const page = await browser.newPage({viewport:{width,height}});
      await page.goto(pathToFileURL(path.join(folder,'midterm-report.html')).href);
      const state = await page.evaluate(() => ({title:document.title,
        width:innerWidth,scrollWidth:document.documentElement.scrollWidth,
        links:[...document.querySelectorAll('a[href]')].map(a=>a.href)}));
      const missing = state.links.filter(url=>url.startsWith('file:') && !fs.existsSync(fileURLToPath(url)));
      await page.screenshot({path:path.join(folder,`preview-${name}.png`)});
      checks.push({name,title:state.title,width:state.width,scrollWidth:state.scrollWidth,
        links:state.links.length,missing,passed:state.scrollWidth<=state.width && missing.length===0});
      await page.close();
    }
    fs.writeFileSync(path.join(folder,'layout-checks.json'),JSON.stringify(checks,null,2));
    console.log(JSON.stringify(checks));
    if (checks.some(x=>!x.passed)) process.exitCode=1;
  } finally { await browser.close(); }
})();
